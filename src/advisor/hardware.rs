use std::process::Command;

use crate::Result;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HardwareProfile {
    pub source: String,
    pub cuda_visible_devices: Option<String>,
    pub gpus: Vec<GpuInfo>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub compute_capability: Option<String>,
    pub memory_total_mib: u64,
    pub memory_free_mib: u64,
    pub memory_used_mib: u64,
}

pub fn format_hardware_profile(profile: &HardwareProfile) -> String {
    let mut text = format!("source: {}\n", profile.source);
    if let Some(value) = &profile.cuda_visible_devices {
        text.push_str(&format!("cuda_visible_devices: {value}\n"));
    }
    text.push_str(&format!("gpus: {}\n", profile.gpus.len()));
    for gpu in &profile.gpus {
        text.push_str(&format!(
            "gpu[{}]: name=\"{}\" uuid={} compute_capability={} memory_total_mib={} memory_free_mib={} memory_used_mib={}\n",
            gpu.index,
            gpu.name,
            gpu.uuid,
            gpu.compute_capability.as_deref().unwrap_or("unknown"),
            gpu.memory_total_mib,
            gpu.memory_free_mib,
            gpu.memory_used_mib,
        ));
    }
    for warning in &profile.warnings {
        text.push_str(&format!("warning: {warning}\n"));
    }
    text
}

pub fn summarize_hardware(profile: &HardwareProfile) -> String {
    if profile.gpus.is_empty() {
        let warning = if profile.warnings.is_empty() {
            "none".to_string()
        } else {
            profile.warnings.join("; ")
        };
        return format!("gpus=0 source={} warning={warning}", profile.source);
    }

    let names = profile
        .gpus
        .iter()
        .map(|gpu| gpu.name.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let total_mib = profile
        .gpus
        .iter()
        .map(|gpu| gpu.memory_total_mib)
        .sum::<u64>();
    let free_mib = profile
        .gpus
        .iter()
        .map(|gpu| gpu.memory_free_mib)
        .sum::<u64>();
    format!(
        "gpus={} names={} memory_total_mib={} memory_free_mib={}",
        profile.gpus.len(),
        names,
        total_mib,
        free_mib
    )
}

pub fn detect_hardware() -> HardwareProfile {
    let mut profile = HardwareProfile {
        source: "nvidia-smi".to_string(),
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .filter(|value| !value.trim().is_empty()),
        gpus: Vec::new(),
        warnings: Vec::new(),
    };

    match query_nvidia_smi(true).and_then(|text| parse_nvidia_smi_csv(&text, true)) {
        Ok(gpus) => profile.gpus = gpus,
        Err(with_compute_cap_error) => {
            match query_nvidia_smi(false).and_then(|text| parse_nvidia_smi_csv(&text, false)) {
                Ok(gpus) => {
                    profile.gpus = gpus;
                    profile.warnings.push(format!(
                        "compute capability unavailable: {with_compute_cap_error}"
                    ));
                }
                Err(error) => profile.warnings.push(error),
            }
        }
    }

    if profile.gpus.is_empty() && profile.warnings.is_empty() {
        profile
            .warnings
            .push("nvidia-smi reported no GPUs".to_string());
    }
    profile
}

fn query_nvidia_smi(include_compute_cap: bool) -> Result<String> {
    let fields = if include_compute_cap {
        "index,name,uuid,compute_cap,memory.total,memory.free,memory.used"
    } else {
        "index,name,uuid,memory.total,memory.free,memory.used"
    };
    let output = Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits".to_string(),
        ])
        .output()
        .map_err(|err| format!("nvidia-smi unavailable: {err}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn parse_nvidia_smi_csv(text: &str, has_compute_cap: bool) -> Result<Vec<GpuInfo>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_gpu_line(line, has_compute_cap))
        .collect()
}

fn parse_gpu_line(line: &str, has_compute_cap: bool) -> Result<GpuInfo> {
    let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
    let expected = if has_compute_cap { 7 } else { 6 };
    if parts.len() != expected {
        return Err(format!(
            "unexpected nvidia-smi row shape: expected {expected} columns, got {}",
            parts.len()
        ));
    }

    let memory_offset = if has_compute_cap { 4 } else { 3 };
    Ok(GpuInfo {
        index: parse_u32(parts[0], "gpu index")?,
        name: parts[1].to_string(),
        uuid: parts[2].to_string(),
        compute_capability: has_compute_cap.then(|| optional(parts[3])).flatten(),
        memory_total_mib: parse_u64(parts[memory_offset], "memory.total")?,
        memory_free_mib: parse_u64(parts[memory_offset + 1], "memory.free")?,
        memory_used_mib: parse_u64(parts[memory_offset + 2], "memory.used")?,
    })
}

fn optional(value: &str) -> Option<String> {
    match value {
        "" | "N/A" | "[Not Supported]" => None,
        value => Some(value.to_string()),
    }
}

fn parse_u32(value: &str, label: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|_| format!("{label} has invalid value: {value}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .split_whitespace()
        .next()
        .unwrap_or(value)
        .parse()
        .map_err(|_| format!("{label} has invalid value: {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_rows_with_compute_capability() {
        let gpus = parse_nvidia_smi_csv(
            "0, NVIDIA L4, GPU-abc, 8.9, 23034, 21000, 2034\n1, NVIDIA L4, GPU-def, 8.9, 23034, 23000, 34\n",
            true,
        )
        .unwrap();

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].name, "NVIDIA L4");
        assert_eq!(gpus[0].compute_capability.as_deref(), Some("8.9"));
        assert_eq!(gpus[0].memory_total_mib, 23034);
        assert_eq!(gpus[0].memory_free_mib, 21000);
        assert_eq!(gpus[0].memory_used_mib, 2034);
    }

    #[test]
    fn parses_nvidia_smi_rows_without_compute_capability() {
        let gpus =
            parse_nvidia_smi_csv("0, NVIDIA A10G, GPU-abc, 23028, 22000, 1028\n", false).unwrap();

        assert_eq!(gpus[0].compute_capability, None);
        assert_eq!(gpus[0].memory_total_mib, 23028);
    }

    #[test]
    fn formats_hardware_profile() {
        let profile = HardwareProfile {
            source: "nvidia-smi".to_string(),
            cuda_visible_devices: Some("0,1".to_string()),
            gpus: vec![GpuInfo {
                index: 0,
                name: "NVIDIA L4".to_string(),
                uuid: "GPU-abc".to_string(),
                compute_capability: Some("8.9".to_string()),
                memory_total_mib: 23034,
                memory_free_mib: 22000,
                memory_used_mib: 1034,
            }],
            warnings: vec!["test warning".to_string()],
        };

        let text = format_hardware_profile(&profile);

        assert!(text.contains("source: nvidia-smi"));
        assert!(text.contains("cuda_visible_devices: 0,1"));
        assert!(text.contains("gpus: 1"));
        assert!(text.contains("compute_capability=8.9"));
        assert!(text.contains("memory_total_mib=23034"));
        assert!(text.contains("warning: test warning"));
    }
}
