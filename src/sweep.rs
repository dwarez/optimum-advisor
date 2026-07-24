use std::collections::{BTreeMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct TrialAllocation {
    pub gpu_devices: Vec<String>,
    pub host_port: u16,
    pub lane: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TrialDemand {
    pub index: usize,
    pub gpus: usize,
}

pub(crate) struct LeaseScheduler {
    devices: Vec<String>,
    ports: Vec<u16>,
    cap: usize,
    pending: Vec<TrialDemand>,
    active: BTreeMap<usize, TrialAllocation>,
}

impl LeaseScheduler {
    pub(crate) fn new(
        devices: Vec<String>,
        ports: Vec<u16>,
        cap: Option<usize>,
        demands: Vec<TrialDemand>,
    ) -> Result<Self> {
        validate_devices(&devices)?;
        validate_ports(&ports)?;
        if cap == Some(0) {
            return Err(Error::validation(
                "sweep.max_parallel_trials must be greater than zero",
            ));
        }
        for (expected, demand) in demands.iter().enumerate() {
            if demand.index != expected {
                return Err(Error::validation(format!(
                    "trial demand index {} is out of order; expected {expected}",
                    demand.index
                )));
            }
            if demand.gpus == 0 || demand.gpus > devices.len() {
                return Err(Error::validation(format!(
                    "trial {} requests {} GPUs from a pool of {}",
                    demand.index,
                    demand.gpus,
                    devices.len()
                )));
            }
        }
        let cap = cap
            .unwrap_or(devices.len())
            .min(devices.len())
            .min(ports.len());
        Ok(Self {
            devices,
            ports,
            cap,
            pending: demands,
            active: BTreeMap::new(),
        })
    }

    pub(crate) fn next(&mut self) -> Option<(usize, TrialAllocation)> {
        if self.active.len() >= self.cap {
            return None;
        }
        let free_devices = self
            .devices
            .iter()
            .filter(|device| {
                !self
                    .active
                    .values()
                    .any(|allocation| allocation.gpu_devices.contains(device))
            })
            .cloned()
            .collect::<Vec<_>>();
        let position = self
            .pending
            .iter()
            .position(|demand| demand.gpus <= free_devices.len())?;
        let demand = self.pending.remove(position);
        let (lane, host_port) = self.ports.iter().copied().enumerate().find(|(lane, _)| {
            !self
                .active
                .values()
                .any(|allocation| allocation.lane == *lane)
        })?;
        let allocation = TrialAllocation {
            gpu_devices: free_devices.into_iter().take(demand.gpus).collect(),
            host_port,
            lane,
        };
        self.active.insert(demand.index, allocation.clone());
        Some((demand.index, allocation))
    }

    pub(crate) fn release(&mut self, index: usize) -> Result<TrialAllocation> {
        self.active.remove(&index).ok_or_else(|| {
            Error::validation(format!("trial {index} does not hold an active GPU lease"))
        })
    }

    pub(crate) fn update_host_port(&mut self, index: usize, host_port: u16) -> Result<()> {
        if host_port == 0
            || self.active.iter().any(|(active_index, allocation)| {
                *active_index != index && allocation.host_port == host_port
            })
        {
            return Err(Error::validation(
                "active sweep host ports must be unique and greater than zero",
            ));
        }
        let allocation = self.active.get_mut(&index).ok_or_else(|| {
            Error::validation(format!("trial {index} does not hold an active GPU lease"))
        })?;
        allocation.host_port = host_port;
        Ok(())
    }

    pub(crate) fn active_host_ports_except(&self, index: usize) -> impl Iterator<Item = u16> + '_ {
        self.active
            .iter()
            .filter(move |(active_index, _)| **active_index != index)
            .map(|(_, allocation)| allocation.host_port)
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn active_len(&self) -> usize {
        self.active.len()
    }
}

fn validate_devices(devices: &[String]) -> Result<()> {
    if devices.is_empty() {
        return Err(Error::validation("sweep GPU pool must not be empty"));
    }
    let mut seen = HashSet::with_capacity(devices.len());
    for device in devices {
        if device.trim().is_empty() || !seen.insert(device) {
            return Err(Error::validation(
                "sweep GPU pool must contain unique nonempty device IDs",
            ));
        }
    }
    Ok(())
}

fn validate_ports(ports: &[u16]) -> Result<()> {
    if ports.is_empty() {
        return Err(Error::validation(
            "sweep requires at least one available host port",
        ));
    }
    let mut seen = HashSet::with_capacity(ports.len());
    if ports.iter().any(|port| *port == 0 || !seen.insert(*port)) {
        return Err(Error::validation(
            "sweep host ports must be unique and greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn devices(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("GPU-{index}")).collect()
    }

    fn demands(values: &[usize]) -> Vec<TrialDemand> {
        values
            .iter()
            .enumerate()
            .map(|(index, gpus)| TrialDemand { index, gpus: *gpus })
            .collect()
    }

    #[test]
    fn backfills_the_first_pending_trial_that_fits() {
        let mut scheduler = LeaseScheduler::new(
            devices(4),
            vec![8_000, 8_001, 8_002, 8_003],
            None,
            demands(&[1, 4, 1]),
        )
        .unwrap();

        let (first_index, first) = scheduler.next().unwrap();
        let (second_index, second) = scheduler.next().unwrap();

        assert_eq!(first_index, 0);
        assert_eq!(first.gpu_devices, vec!["GPU-0"]);
        assert_eq!(second_index, 2);
        assert_eq!(second.gpu_devices, vec!["GPU-1"]);
        assert!(scheduler.next().is_none());
        assert!(scheduler.has_pending());

        scheduler.release(first_index).unwrap();
        scheduler.release(second_index).unwrap();
        let (last_index, last) = scheduler.next().unwrap();
        assert_eq!(last_index, 1);
        assert_eq!(last.gpu_devices, devices(4));
        assert_eq!(last.lane, 0);
        assert_eq!(last.host_port, 8_000);
    }

    #[test]
    fn leases_requested_count_in_configured_device_order() {
        let mut scheduler =
            LeaseScheduler::new(devices(4), vec![8_000], None, demands(&[3])).unwrap();

        let (_, allocation) = scheduler.next().unwrap();

        assert_eq!(allocation.gpu_devices, devices(3));
    }

    #[test]
    fn explicit_cap_serializes_otherwise_schedulable_trials() {
        let mut scheduler =
            LeaseScheduler::new(devices(2), vec![8_000, 8_001], Some(1), demands(&[1, 1])).unwrap();

        let (first, allocation) = scheduler.next().unwrap();
        assert!(scheduler.next().is_none());
        assert_eq!(scheduler.active_len(), 1);

        scheduler.release(first).unwrap();
        assert_eq!(scheduler.next().unwrap().0, 1);
        assert_eq!(allocation.gpu_devices, vec!["GPU-0"]);
    }

    #[test]
    fn never_reuses_an_active_device_or_lane() {
        let mut scheduler = LeaseScheduler::new(
            devices(4),
            vec![8_000, 8_001, 8_002],
            None,
            demands(&[2, 1, 1]),
        )
        .unwrap();

        let allocations = (0..3)
            .map(|_| scheduler.next().unwrap().1)
            .collect::<Vec<_>>();

        for (left_index, left) in allocations.iter().enumerate() {
            for right in allocations.iter().skip(left_index + 1) {
                assert!(left
                    .gpu_devices
                    .iter()
                    .all(|device| !right.gpu_devices.contains(device)));
                assert_ne!(left.lane, right.lane);
                assert_ne!(left.host_port, right.host_port);
            }
        }
    }

    #[test]
    fn validates_resource_and_demand_invariants() {
        assert!(LeaseScheduler::new(Vec::new(), vec![8_000], None, demands(&[1])).is_err());
        assert!(LeaseScheduler::new(devices(1), Vec::new(), None, demands(&[1])).is_err());
        assert!(LeaseScheduler::new(devices(1), vec![8_000], Some(0), demands(&[1])).is_err());
        assert!(LeaseScheduler::new(devices(1), vec![8_000], None, demands(&[2])).is_err());
        assert!(LeaseScheduler::new(
            vec!["GPU-0".into(), "GPU-0".into()],
            vec![8_000],
            None,
            demands(&[1]),
        )
        .is_err());
        assert!(LeaseScheduler::new(devices(1), vec![8_000, 8_000], None, demands(&[1])).is_err());
    }
}
