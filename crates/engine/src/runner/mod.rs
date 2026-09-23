//! runner 进程协议层(MessageType/RunnerInitRequest/socket 序列化助手)。
//! 属 core:engine(父进程侧)与 runner 进程(server crate)共用同一协议定义。
use crate::sequence::{DecodeSequence, Sequence};
use crate::distributed::Id;
use crate::server::EmbeddingStrategy;
use crate::config::{Config, EngineConfig, ModelType};
use crate::downloader::ModelPaths;
#[cfg(feature = "nccl")]
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use interprocess::local_socket::Stream as LocalStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunnerInitRequest {
    pub rank: usize,
    pub dev_id: usize,
    pub num_shards: usize,
    pub model_type: ModelType,
    pub config: Config,
    pub econfig: EngineConfig,
    pub model_pathes: ModelPaths,
    pub is_gguf: bool,
    pub dtype: SerializableDType,
    pub is_rope_i: bool,
    #[cfg(feature = "nccl")]
    pub nccl_id: NcclId,
}

#[derive(Debug, Clone)]
pub struct NcclId(pub Id);

#[cfg(feature = "nccl")]
impl Serialize for NcclId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Detect if JSON serializer
        if serializer.is_human_readable() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            let encoded = STANDARD_NO_PAD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            serializer.serialize_bytes(bytes)
        }
    }
}

#[cfg(feature = "nccl")]
impl<'de> Deserialize<'de> for NcclId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s: &str = Deserialize::deserialize(deserializer)?;
            let bytes = STANDARD_NO_PAD
                .decode(s)
                .map_err(serde::de::Error::custom)?;
            if bytes.len() != 128 {
                return Err(serde::de::Error::custom(format!(
                    "Expected 128 bytes but got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0i8; 128];
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
            }
            #[cfg(not(target_arch = "aarch64"))]
            return Ok(NcclId(Id::uninit(arr)));
            #[cfg(target_arch = "aarch64")]
            {
                let arr_u8 = arr.map(|b| b as u8);
                return Ok(NcclId(Id::uninit(arr_u8)));
            }
        } else {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = NcclId;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "128-byte NCCL ID")
                }

                fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    if v.len() != 128 {
                        return Err(E::custom(format!("Expected 128 bytes but got {}", v.len())));
                    }
                    let mut arr = [0i8; 128];
                    unsafe {
                        std::ptr::copy_nonoverlapping(v.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    return Ok(NcclId(Id::uninit(arr)));
                    #[cfg(target_arch = "aarch64")]
                    {
                        let arr_u8 = arr.map(|b| b as u8);
                        return Ok(NcclId(Id::uninit(arr_u8)));
                    }
                }
            }

            deserializer.deserialize_bytes(Visitor)
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum SerializableDType {
    U8 = 0,
    U32 = 1,
    I64 = 2,
    BF16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
    F8E8M0 = 7,
    F8E4M3 = 8,
}

impl From<owl_nn::Dtype> for SerializableDType {
    fn from(dt: owl_nn::Dtype) -> Self {
        match dt {
            owl_nn::Dtype::U8 => Self::U8,
            owl_nn::Dtype::U32 => Self::U32,
            owl_nn::Dtype::I64 => Self::I64,
            owl_nn::Dtype::BF16 => Self::BF16,
            owl_nn::Dtype::F16 => Self::F16,
            owl_nn::Dtype::F32 => Self::F32,
            owl_nn::Dtype::F8E8M0 => Self::F8E8M0,
            // 量化占位无对应 wire 变体(线协议保持 xinfer 原样)
            owl_nn::Dtype::Q8 => unimplemented!("T3 搬运期回填:Q8 wire 语义待定"),
        }
    }
}

impl From<SerializableDType> for owl_nn::Dtype {
    fn from(sdt: SerializableDType) -> Self {
        match sdt {
            SerializableDType::U8 => owl_nn::Dtype::U8,
            SerializableDType::U32 => owl_nn::Dtype::U32,
            SerializableDType::I64 => owl_nn::Dtype::I64,
            SerializableDType::BF16 => owl_nn::Dtype::BF16,
            SerializableDType::F16 => owl_nn::Dtype::F16,
            SerializableDType::F32 => owl_nn::Dtype::F32,
            SerializableDType::F8E8M0 => owl_nn::Dtype::F8E8M0,
            // owl 设备路径无 f64 / f8e4m3(T3 搬运期回填)
            SerializableDType::F64 => unimplemented!("T3 搬运期回填:owl 无 f64 设备路径"),
            SerializableDType::F8E4M3 => unimplemented!("T3 搬运期回填:owl 无 f8e4m3 设备路径"),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct InitAck {
    pub ok: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    /// Sent by main process to initialize the runner.
    Init(RunnerInitRequest),

    /// Sent by runner in response to `Init` with initialization status.
    InitAck(bool),

    LoadingProgress((usize, usize)),

    /// Sent by main process to request prefill on sequences.
    RunPrefill((Vec<Sequence>, bool)),

    /// Sent by main process to request inference on sequences.
    RunDecode((Vec<DecodeSequence>, bool)),

    /// Sent by main process to request MTP speculative decode on sequences.
    RunDecodeMTP(Vec<DecodeSequence>),

    /// Sent by runner in response to `Run` with generated token IDs.
    RunResponse(Vec<u32>),

    /// Sent by runner in response to `RunDecodeMTP` with multiple tokens per sequence.
    RunResponseMTP(Vec<Vec<u32>>),
    /// Sent by the main process to request external DFlash speculative decode.
    RunDecodeDFlash(Vec<DecodeSequence>),
    /// Sent by a runner in response to `RunDecodeDFlash`.
    RunResponseDFlash(Vec<Vec<u32>>),

    /// Sent by main process to request embedding on sequences.
    RunEmbed((Vec<Sequence>, EmbeddingStrategy)),

    /// Sent by runner in response to `Run` with generated embedding states
    RunResponseEmbed(Vec<Vec<f32>>),

    /// Sent by main process to notify the finished decoding sequences.
    FinishDecode(usize),

    // Hybrid mamba-prefix state management.
    CaptureMambaPrefixState((usize, u64, bool)),
    CaptureMambaPrefixStateResponse(bool),
    HasMambaPrefixState(u64),
    HasMambaPrefixStateResponse(bool),
    RemoveMambaPrefixState(u64),
    RemoveMambaPrefixStateResponse(bool),

    /// Optional: runner can send back an error message.
    Error(String),

    Heartbeat,

    // Prefill transfer to PD server
    TransferPrefill(Sequence),
    TransferPrefillResponse(bool),

    // Prefill transfer receive
    ReceivePrefill(usize),
    ReceivePrefillResponse((bool, Option<Sequence>)),

    // Client: Check PD prefill status
    CheckPrefillStatus(usize),
    CheckPrefillStatusResponse(bool),

    RunSpecDecode(Vec<Sequence>),
    RunDraftAndVerify((Vec<Sequence>, Vec<u32>)),

    RunSpecDecodeResponse(Vec<Vec<u32>>),

    KVCacheSwap((HashMap<usize, usize>, bool)),

    KVCacheSwapResponse(bool),

    // send kvcache to client (seq_id, first_token)
    KvCacheSend((Sequence, u32)),
    KvCacheSendResponse(bool),

    // receive kvcache from PD server
    KvCacheReceive(Sequence),
    KvCacheReceiveResponse((bool, u32, usize, usize)),

    // notify PD server to release kvcache
    KvCacheRelease(usize),
    KvCacheReleaseResponse(bool),

    // Server: Check if a prefilled seq need to release kvcache
    CheckKvCacheRelease(usize),
    CheckKvCacheReleaseResponse(bool),

    ClearBlocks(Vec<u32>),
    ClearBlocksResponse(bool),

    UsableMemoryLeft(EngineConfig),
    /// shutdown subprocesses
    Shutdown,
}

//inter-node communication
pub fn send_local(
    streams: &mut Vec<LocalStream>,
    message: &MessageType,
    use_json: bool,
) -> std::io::Result<()> {
    let serialized = if use_json {
        serde_json::to_vec(message).expect("JSON serialization failed")
    } else {
        bincode::serialize(message).expect("Bincode serialization failed")
    };

    for stream in streams.iter_mut() {
        stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
        stream.write_all(&serialized)?;
        stream.flush()?; // Ensure data is sent immediately
                         // Wait for acknowledgment
        let mut ack_buf = [0u8; 1];
        if let Err(e) = stream.read_exact(&mut ack_buf) {
            eprintln!(
                "Timeout waiting for acknowledgment from subprocess: {:?}",
                e
            );
        } else if ack_buf[0] != 1 {
            eprintln!("Unexpected acknowledgment value from subprocess");
        }
    }
    Ok(())
}

pub fn receive_local(stream: &mut LocalStream, use_json: bool) -> std::io::Result<MessageType> {
    let mut length_buf = [0u8; 4];
    stream.read_exact(&mut length_buf)?;
    let length = u32::from_le_bytes(length_buf) as usize;

    let mut serialized = vec![0u8; length];
    stream.read_exact(&mut serialized)?;

    let message: MessageType = if use_json {
        serde_json::from_slice(&serialized).map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSON deserialization failed: {err}"),
            )
        })?
    } else {
        bincode::deserialize(&serialized).map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Bincode deserialization failed: {err}"),
            )
        })?
    };

    // Send acknowledgment
    stream.write_all(&[1])?;
    stream.flush()?;
    Ok(message)
}

pub fn send_and_expect_ack(
    stream: &mut LocalStream,
    msg: &MessageType,
    stage: &str,
    rank: usize,
) -> crate::Result<()> {
    use interprocess::TryClone;
    send_local(&mut vec![stream.try_clone()?], msg, true)?;

    crate::log_info!("Waiting runner {} {} response...", rank, stage);

    match receive_local(stream, false)? {
        MessageType::InitAck(true) => Ok(()),
        _ => crate::bail!("Runner {} failed during {}", rank, stage),
    }
}

///
/// Defines a function that broadcasts an operation to all runners and expects a `Result<T>`.
/// It handles both `Thread` (direct call) and `Process` (IPC message) runners.
///
/// In Process mode, it expects the response variant to contain the value `T`.
/// It collects all values and verifies they are identical before returning one.
///
#[macro_export]
macro_rules! def_broadcast_message_to_runners {
    (
        // The visibility (e.g., `pub`)
        $vis:vis,
        // The name of the function to create (e.g., `try_receive_kv_cache`)
        $fn_name:ident,
        // The name of the method on the thread-mode runner (e.g., `receive_kv_cache`)
        $thread_fn_name:ident,
        // The arguments for the function (e.g., `(seq: Sequence)`)
        ($($arg_name:ident: $arg_type:ty),*),
        // The MessageType variant to send (e.g., `MessageType::KvCacheReceive`)
        $msg_variant:path,
        // The expression to build the message payload (e.g., `(seq.clone())`)
        ($($msg_arg:expr),*),
        // The MessageType response variant to match (e.g., `MessageType::KvCacheReceiveResponse`)
        $resp_variant:path,
        // The inner return type (e.g., `u32`)
        $return_ty:ty
    ) => {
        $vis fn $fn_name(&self, $($arg_name: $arg_type),*) -> Result<$return_ty>
        where
            $return_ty: std::fmt::Debug + Send,
        {
            match &mut *self.runners.write() {
                RunnerType::Thread(model_runner) => {
                    // Thread Mode: Call the method directly.
                    model_runner.$thread_fn_name($($arg_name),*)
                }
                RunnerType::Process(ref mut runner_streams) => {
                    // Process Mode: Broadcast to all subprocess runners.
                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s| s.try_clone().expect("Failed to clone runner stream"))
                        .collect();

                    // Use Rayon for parallel broadcast
                    let all_results: Result<Vec<$return_ty>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            // Send the message
                            send_local(
                                &mut vec![stream.try_clone()?],
                                &$msg_variant($($msg_arg),*),
                                false,
                            )?;

                            // Wait for the response
                            let response = receive_local(&mut stream, false)?;
                            match response {
                                // Match on the expected response containing the value
                                $resp_variant(value) => {
                                    Ok(value)
                                }
                                other => {
                                    crate::bail!("Unexpected response for {}: {:?}", stringify!($fn_name), other)
                                }
                            }
                        })
                        .collect(); // Collects into a Result<Vec<T>>

                    // Check that all ranks returned the same value
                    match all_results {
                        Ok(mut values) => {
                            if values.is_empty() {
                                crate::bail!("No values received from runners for {}", stringify!($fn_name));
                            }
                            // Pop first element to return, then check rest for consistency
                            let first_val = values.pop().unwrap();
                            Ok(first_val)
                        }
                        Err(e) => Err(e),
                    }
                }
                RunnerType::MultiNodeMaster {
                    local_streams: ref mut runner_streams,
                    remote_streams: ref mut remote_streams,
                } => {
                    let request = $msg_variant($($msg_arg),*);
                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s| s.try_clone().expect("Failed to clone runner stream"))
                        .collect();

                    let local_results: Result<Vec<$return_ty>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            let msg = request.clone();
                            send_local(&mut vec![stream.try_clone()?], &msg, false)?;

                            let response = receive_local(&mut stream, false)?;
                            match response {
                                $resp_variant(value) => Ok(value),
                                other => {
                                    crate::bail!("Unexpected local response for {}: {:?}", stringify!($fn_name), other)
                                }
                            }
                        })
                        .collect();

                    let mut values = local_results?;
                    let serialized = bincode::serialize(&request).expect("Bincode serialization failed");

                    for tcp_stream in remote_streams.iter_mut() {
                        crate::multi_node::send_tcp(tcp_stream, &serialized)?;
                    }

                    for tcp_stream in remote_streams.iter_mut() {
                        let data = crate::multi_node::recv_tcp(tcp_stream)?;
                        let response: MessageType = bincode::deserialize(&data)
                            .expect("Bincode deserialization failed");
                        match response {
                            $resp_variant(value) => values.push(value),
                            MessageType::Error(err) => {
                                crate::bail!("Remote worker failed for {}: {}", stringify!($fn_name), err)
                            }
                            other => {
                                crate::bail!("Unexpected remote response for {}: {:?}", stringify!($fn_name), other)
                            }
                        }
                    }

                    if values.is_empty() {
                        crate::bail!("No values received from runners for {}", stringify!($fn_name));
                    }
                    Ok(values.remove(0))
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// T3 占位:线程模式 runner 与 RunnerType(自 xinfer `core/runner.rs:166` 搬运
// 枚举本身;ModelRunner 是 T3 大件——整条模型/采样/KV/投机链,编译先行口径
// 下以 12 个宏触点的签名 stub 表达,方法体 T3 回填)。
// ---------------------------------------------------------------------------

/// T3 占位:线程模式模型 runner(xinfer `ModelRunner`)。
///
/// 12 个方法 = `def_broadcast_message_to_runners!` 的 Thread 分支调用面
/// (block_manager 12 个触点);签名与 xinfer 一一对应,方法体 T3 回填。
#[allow(dead_code)] // T3 回填前的占位面
pub struct ModelRunner;

impl ModelRunner {
    pub fn transfer_prefill(&self, _seq: &crate::sequence::Sequence) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn try_receive_prefill(
        &self,
        _available_tokens: usize,
    ) -> crate::Result<(bool, Option<crate::sequence::Sequence>)> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn check_prefill_status(&self, _seq_id: usize) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn swap_kvcache(
        &self,
        _mappings: std::collections::HashMap<usize, usize>,
        _swap_in: bool,
    ) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn send_kvcache(
        &self,
        _seq: &crate::sequence::Sequence,
        _token: u32,
    ) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn receive_kvcache(
        &self,
        _seq: &crate::sequence::Sequence,
    ) -> crate::Result<(bool, u32, usize, usize)> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn release_remote_kvcache(&self, _seq_id: usize) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn check_kvcache_release(&self, _seq_id: usize) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn capture_mamba_prefix_state(
        &self,
        _seq_id: usize,
        _hash: u64,
        _preserve: bool,
    ) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn has_mamba_prefix_state(&self, _hash: u64) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn remove_mamba_prefix_state(&self, _hash: u64) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
    pub fn clear_blocks(&self, _block_ids: Vec<u32>) -> crate::Result<bool> {
        unimplemented!("T3 搬运期回填")
    }
}

/// runner 运行形态(xinfer `RunnerType` 原样;Thread 载荷为 T3 占位)。
pub enum RunnerType {
    Thread(ModelRunner),
    Process(Vec<LocalStream>),
    /// Master node in multi-node inference: local IPC streams + TCP streams to worker nodes.
    MultiNodeMaster {
        local_streams: Vec<LocalStream>,
        remote_streams: Vec<std::net::TcpStream>,
    },
}

