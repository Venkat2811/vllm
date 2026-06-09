use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

use bytes::Bytes;
use myelon::portable_shm_segment_name;
use myelon::transport::{FixedFrame, FramedTransportProducer};
use myelon_zmq::SendSocket;
use tokio::sync::mpsc;
use zeromq::prelude::{Socket, SocketSend};
use zeromq::{PullSocket, PushSocket, ZmqMessage};

use vllm_engine_core_client::myelon_transport::{
    FRAME_BYTES, MyelonNarrowOutputSocket, MyelonOutputSocket, RING_SLOTS,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, encode_msgpack,
};
use vllm_engine_core_client::{ENGINE_CORE_DEAD_SENTINEL, OutputSource, Result, run_output_loop};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Zmq,
    MyelonMultipart,
    MyelonNarrow,
}

impl Mode {
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw {
            "zmq" => Ok(Self::Zmq),
            "myelon" | "multipart" => Ok(Self::MyelonMultipart),
            "narrow" => Ok(Self::MyelonNarrow),
            other => Err(format!("unsupported mode `{other}`")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Zmq => "zmq",
            Self::MyelonMultipart => "myelon_multipart",
            Self::MyelonNarrow => "myelon_narrow",
        }
    }
}

#[derive(Debug)]
struct Config {
    mode: Option<Mode>,
    messages: usize,
    target_bytes: usize,
    warmup_messages: usize,
    channel_capacity: usize,
}

impl Config {
    fn parse() -> std::result::Result<Self, String> {
        let mut mode = None;
        let mut messages = 200_000usize;
        let mut target_bytes = 4096usize;
        let mut warmup_messages = 5_000usize;
        let mut channel_capacity = 4096usize;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => mode = Some(Mode::parse(&next_arg(&mut args, "--mode")?)?),
                "--messages" => {
                    messages = parse_usize(&next_arg(&mut args, "--messages")?, "--messages")?
                }
                "--target-bytes" => {
                    target_bytes =
                        parse_usize(&next_arg(&mut args, "--target-bytes")?, "--target-bytes")?
                }
                "--warmup-messages" => {
                    warmup_messages = parse_usize(
                        &next_arg(&mut args, "--warmup-messages")?,
                        "--warmup-messages",
                    )?
                }
                "--channel-capacity" => {
                    channel_capacity = parse_usize(
                        &next_arg(&mut args, "--channel-capacity")?,
                        "--channel-capacity",
                    )?
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(format!("unexpected arg `{other}`")),
            }
        }

        Ok(Self {
            mode,
            messages,
            target_bytes,
            warmup_messages,
            channel_capacity,
        })
    }
}

#[derive(Debug)]
struct PayloadFixture {
    bytes: Vec<u8>,
    output_items: usize,
}

#[derive(Debug)]
struct BenchResult {
    mode: Mode,
    messages: usize,
    payload_bytes: usize,
    output_items: usize,
    elapsed: Duration,
}

impl BenchResult {
    fn print(&self) {
        let secs = self.elapsed.as_secs_f64();
        let msgps = self.messages as f64 / secs;
        let mibps = (self.messages * self.payload_bytes) as f64 / secs / (1024.0 * 1024.0);
        println!(
            "RESULT mode={} messages={} payload_bytes={} output_items={} elapsed_s={:.6} msgps={:.2} mibps={:.2}",
            self.mode.as_str(),
            self.messages,
            self.payload_bytes,
            self.output_items,
            secs,
            msgps,
            mibps
        );
    }
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let config = Config::parse()?;
    let fixture = build_payload(config.target_bytes)?;

    println!(
        "PAYLOAD actual_payload_bytes={} output_items={} target_bytes={} messages={} warmup_messages={} channel_capacity={}",
        fixture.bytes.len(),
        fixture.output_items,
        config.target_bytes,
        config.messages,
        config.warmup_messages,
        config.channel_capacity
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(16 * 1024 * 1024)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        match config.mode {
            Some(mode) => {
                let result = run_one(mode, &fixture, &config).await?;
                result.print();
            }
            None => {
                for mode in [Mode::Zmq, Mode::MyelonMultipart, Mode::MyelonNarrow] {
                    let result = run_one(mode, &fixture, &config).await?;
                    result.print();
                }
            }
        }
        Ok::<(), Box<dyn Error>>(())
    })?;

    Ok(())
}

async fn run_one(
    mode: Mode,
    fixture: &PayloadFixture,
    config: &Config,
) -> std::result::Result<BenchResult, Box<dyn Error>> {
    match mode {
        Mode::Zmq => run_zmq_bench(fixture, config).await,
        Mode::MyelonMultipart => run_myelon_multipart_bench(fixture, config).await,
        Mode::MyelonNarrow => run_myelon_narrow_bench(fixture, config).await,
    }
}

async fn run_zmq_bench(
    fixture: &PayloadFixture,
    config: &Config,
) -> std::result::Result<BenchResult, Box<dyn Error>> {
    let endpoint = unique_ipc_endpoint("z");
    let mut pull = PullSocket::new();
    let bound = pull.bind(&endpoint).await?.to_string();
    let mut push = PushSocket::new();
    push.connect(&bound).await?;

    let (tx, mut rx) = mpsc::channel::<Result<EngineCoreOutputs>>(config.channel_capacity);
    let output_task = tokio::spawn(run_output_loop(OutputSource::Zmq(pull), tx));

    send_warmup_zmq(&mut push, &fixture.bytes, config.warmup_messages).await?;
    drain_successes(&mut rx, config.warmup_messages).await?;

    let payload = Bytes::from(fixture.bytes.clone());
    let messages = config.messages;
    let start = Instant::now();
    let sender = tokio::spawn(async move {
        for _ in 0..messages {
            push.send(ZmqMessage::from(payload.clone()))
                .await
                .map_err(|err| err.to_string())?;
        }
        push.send(ZmqMessage::from(Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)))
            .await
            .map_err(|err| err.to_string())?;
        Ok::<(), String>(())
    });

    drain_successes(&mut rx, config.messages).await?;
    let elapsed = start.elapsed();

    sender
        .await
        .map_err(|err| format!("zmq sender join failed: {err}"))?
        .map_err(|err| format!("zmq sender failed: {err}"))?;
    drop(rx);
    output_task.await?;

    Ok(BenchResult {
        mode: Mode::Zmq,
        messages: config.messages,
        payload_bytes: fixture.bytes.len(),
        output_items: fixture.output_items,
        elapsed,
    })
}

async fn run_myelon_multipart_bench(
    fixture: &PayloadFixture,
    config: &Config,
) -> std::result::Result<BenchResult, Box<dyn Error>> {
    let segment = portable_shm_segment_name("rpym");
    let mut producer: SendSocket<FRAME_BYTES> = SendSocket::bind(&segment, RING_SLOTS, "vlm_0")?;

    let recv_segment = segment.clone();
    let recv = tokio::task::spawn_blocking(move || MyelonOutputSocket::connect_to_segment(&recv_segment))
        .await??;
    producer.warm_discovery(Duration::from_secs(2));

    let (tx, mut rx) = mpsc::channel::<Result<EngineCoreOutputs>>(config.channel_capacity);
    let output_task = tokio::spawn(run_output_loop(OutputSource::Myelon(recv), tx));

    send_warmup_multipart(&mut producer, &fixture.bytes, config.warmup_messages)?;
    drain_successes(&mut rx, config.warmup_messages).await?;

    let payload = fixture.bytes.clone();
    let messages = config.messages;
    let start = Instant::now();
    let sender = tokio::task::spawn_blocking(move || {
        for _ in 0..messages {
            producer
                .send_multipart(&[payload.as_slice()])
                .map_err(|err| err.to_string())?;
        }
        producer
            .send_multipart(&[ENGINE_CORE_DEAD_SENTINEL])
            .map_err(|err| err.to_string())?;
        Ok::<(), String>(())
    });

    drain_successes(&mut rx, config.messages).await?;
    let elapsed = start.elapsed();

    sender
        .await
        .map_err(|err| format!("multipart sender join failed: {err}"))?
        .map_err(|err| format!("multipart sender failed: {err}"))?;
    drop(rx);
    output_task.await?;

    Ok(BenchResult {
        mode: Mode::MyelonMultipart,
        messages: config.messages,
        payload_bytes: fixture.bytes.len(),
        output_items: fixture.output_items,
        elapsed,
    })
}

async fn run_myelon_narrow_bench(
    fixture: &PayloadFixture,
    config: &Config,
) -> std::result::Result<BenchResult, Box<dyn Error>> {
    let segment = portable_shm_segment_name("rpyn");
    let mut producer = FramedTransportProducer::<FixedFrame<FRAME_BYTES>>::create(&segment, RING_SLOTS)?;

    let recv_segment = segment.clone();
    let recv = tokio::task::spawn_blocking(move || MyelonNarrowOutputSocket::connect_to_segment(&recv_segment))
        .await??;
    let _ = producer.discover_consumer_id("vlm_0", Duration::from_secs(2));

    let (tx, mut rx) = mpsc::channel::<Result<EngineCoreOutputs>>(config.channel_capacity);
    let output_task = tokio::spawn(run_output_loop(OutputSource::MyelonNarrow(recv), tx));

    send_warmup_narrow(&mut producer, &fixture.bytes, config.warmup_messages);
    drain_successes(&mut rx, config.warmup_messages).await?;

    let payload = fixture.bytes.clone();
    let messages = config.messages;
    let start = Instant::now();
    let sender = tokio::task::spawn_blocking(move || {
        for _ in 0..messages {
            producer.publish(payload.as_slice(), 0);
        }
        producer.publish(ENGINE_CORE_DEAD_SENTINEL, 0);
        Ok::<(), String>(())
    });

    drain_successes(&mut rx, config.messages).await?;
    let elapsed = start.elapsed();

    sender
        .await
        .map_err(|err| format!("narrow sender join failed: {err}"))?
        .map_err(|err| format!("narrow sender failed: {err}"))?;
    drop(rx);
    output_task.await?;

    Ok(BenchResult {
        mode: Mode::MyelonNarrow,
        messages: config.messages,
        payload_bytes: fixture.bytes.len(),
        output_items: fixture.output_items,
        elapsed,
    })
}

async fn drain_successes(
    rx: &mut mpsc::Receiver<Result<EngineCoreOutputs>>,
    expected: usize,
) -> std::result::Result<(), Box<dyn Error>> {
    let mut seen = 0usize;
    while seen < expected {
        let msg = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await?;
        match msg {
            Some(Ok(decoded)) => {
                let _touch = decoded.outputs.len();
                seen += 1;
            }
            Some(Err(err)) => return Err(format!("unexpected output-loop error after {seen} messages: {err}").into()),
            None => return Err(format!("output loop closed after {seen} messages").into()),
        }
    }
    Ok(())
}

async fn send_warmup_zmq(
    push: &mut PushSocket,
    payload: &[u8],
    warmup_messages: usize,
) -> std::result::Result<(), zeromq::ZmqError> {
    let bytes = Bytes::copy_from_slice(payload);
    for _ in 0..warmup_messages {
        push.send(ZmqMessage::from(bytes.clone())).await?;
    }
    Ok(())
}

fn send_warmup_multipart(
    producer: &mut SendSocket<FRAME_BYTES>,
    payload: &[u8],
    warmup_messages: usize,
) -> std::result::Result<(), Box<dyn Error>> {
    for _ in 0..warmup_messages {
        producer.send_multipart(&[payload])?;
    }
    Ok(())
}

fn send_warmup_narrow(
    producer: &mut FramedTransportProducer<FixedFrame<FRAME_BYTES>>,
    payload: &[u8],
    warmup_messages: usize,
) {
    for _ in 0..warmup_messages {
        producer.publish(payload, 0);
    }
}

fn build_payload(target_bytes: usize) -> std::result::Result<PayloadFixture, Box<dyn Error>> {
    let mut outputs = 1usize;
    loop {
        let candidate = make_outputs(outputs);
        let bytes = encode_msgpack(&candidate)?;
        if bytes.len() >= target_bytes {
            return Ok(PayloadFixture {
                bytes,
                output_items: outputs,
            });
        }
        outputs *= 2;
    }
}

fn make_outputs(output_items: usize) -> EngineCoreOutputs {
    let outputs = (0..output_items)
        .map(|i| EngineCoreOutput {
            request_id: format!("req-{i:05}"),
            new_token_ids: vec![((i % 2048) + 1) as u32],
            new_logprobs: None,
            new_prompt_logprobs_tensors: None,
            pooling_output: None,
            finish_reason: if i % 17 == 0 {
                Some(EngineCoreFinishReason::Length)
            } else {
                None
            },
            stop_reason: None,
            events: None,
            kv_transfer_params: None,
            trace_headers: None,
            prefill_stats: None,
            routed_experts: None,
            num_nans_in_logits: 0,
        })
        .collect::<Vec<_>>();

    EngineCoreOutputs {
        engine_index: 0,
        outputs,
        scheduler_stats: None,
        timestamp: 0.0,
        utility_output: None,
        finished_requests: Some(BTreeSet::new()),
        wave_complete: None,
        start_wave: None,
    }
}

fn unique_ipc_endpoint(tag: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("unix epoch")
        .as_nanos() as u64;
    let path = format!("/tmp/{tag}{}{:x}.sock", std::process::id(), nonce & 0xffff_ffff);
    let path_ref = Path::new(&path);
    if path_ref.exists() {
        let _ = std::fs::remove_file(path_ref);
    }
    format!("ipc://{path}")
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> std::result::Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize(raw: &str, flag: &str) -> std::result::Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|err| format!("{flag} expects usize, got `{raw}`: {err}"))
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --example output_transport_replay --features myelon_hot_path -- \
  [--mode zmq|myelon|narrow] \
  [--messages N] [--target-bytes N] [--warmup-messages N] [--channel-capacity N]"
    );
}
