// Restabilește destinațiile fluxurilor Mid360 către serviciile native G1.
// Nu schimbă IP-ul LiDAR-ului, extrinsecii sau locomotia robotului.

#include "livox_lidar_api.h"
#include "livox_lidar_def.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

namespace {

std::atomic<uint32_t> discovered_handle{0};

struct CommandResult {
  std::atomic<bool> done{false};
  std::atomic<bool> success{false};
};

void OnLidar(const uint32_t handle, const LivoxLidarInfo* info, void*) {
  if (info != nullptr) {
    std::printf("[mid360-restore] detectat SN=%s handle=%u\n", info->sn, handle);
    discovered_handle.store(handle);
  }
}

void OnCommand(livox_status status, uint32_t,
               LivoxLidarAsyncControlResponse* response, void* client_data) {
  auto* result = static_cast<CommandResult*>(client_data);
  const bool ok = status == 0 && response != nullptr &&
                  response->ret_code == 0 && response->error_key == 0;
  if (response != nullptr) {
    std::printf("[mid360-restore] status=%d ret_code=%u error_key=%u\n",
                status, response->ret_code, response->error_key);
  } else {
    std::printf("[mid360-restore] status=%d fără răspuns\n", status);
  }
  result->success.store(ok);
  result->done.store(true);
}

bool WaitResult(CommandResult& result, int timeout_seconds = 8) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_seconds);
  while (!result.done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return result.done.load() && result.success.load();
}

bool IssueAndWait(
    const char* label,
    const std::function<livox_status(CommandResult*)>& issue) {
  CommandResult result;
  const livox_status issued = issue(&result);
  if (issued != 0 || !WaitResult(result)) {
    std::fprintf(stderr, "[mid360-restore] %s neconfirmat (send=%d)\n",
                 label, issued);
    return false;
  }
  std::printf("[mid360-restore] %s confirmat\n", label);
  return true;
}

template <typename Config, typename Setter>
bool SetAndWait(const char* label, Config& config, Setter setter) {
  CommandResult result;
  const livox_status issued = setter(
      discovered_handle.load(), &config, OnCommand, &result);
  if (issued != 0 || !WaitResult(result)) {
    std::fprintf(stderr, "[mid360-restore] %s neconfirmat (send=%d)\n",
                 label, issued);
    return false;
  }
  std::printf("[mid360-restore] %s confirmat\n", label);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Utilizare: %s CONFIG_LOCAL IP_UNITREE\n", argv[0]);
    return 2;
  }
  if (!LivoxLidarSdkInit(argv[1])) {
    std::fprintf(stderr, "[mid360-restore] LivoxLidarSdkInit a eșuat\n");
    return 3;
  }
  SetLivoxLidarInfoChangeCallback(OnLidar, nullptr);

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(10);
  while (discovered_handle.load() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (discovered_handle.load() == 0) {
    std::fprintf(stderr, "[mid360-restore] Mid360 nu a fost descoperit\n");
    LivoxLidarSdkUninit();
    return 4;
  }
  const uint32_t handle = discovered_handle.load();

  const bool pcl_ok = IssueAndWait("format Cartesian high", [handle](CommandResult* r) {
    return SetLivoxLidarPclDataType(
        handle, kLivoxLidarCartesianCoordinateHighData, OnCommand, r);
  });
  const bool pattern_ok = IssueAndWait("pattern non-repetitiv", [handle](CommandResult* r) {
    return SetLivoxLidarScanPattern(
        handle, kLivoxLidarScanPatternNoneRepetive, OnCommand, r);
  });
  const bool normal_ok = IssueAndWait("work mode Normal", [handle](CommandResult* r) {
    return SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, OnCommand, r);
  });
  // Firmware-ul G1 poate raporta "unsupported" aici chiar dacă modul Normal
  // livrează punctele. Comanda rămâne diagnostică, nu o condiție de succes.
  IssueAndWait("point send ON", [handle](CommandResult* r) {
    return EnableLivoxLidarPointSend(handle, OnCommand, r);
  });
  const bool imu_send_ok = IssueAndWait("IMU send ON", [handle](CommandResult* r) {
    return EnableLivoxLidarImuData(handle, OnCommand, r);
  });

  HostPointIPInfo point{};
  std::snprintf(point.host_ip_addr, sizeof(point.host_ip_addr), "%s", argv[2]);
  point.host_point_data_port = 56301;
  point.lidar_point_data_port = 56300;

  HostImuDataIPInfo imu{};
  std::snprintf(imu.host_ip_addr, sizeof(imu.host_ip_addr), "%s", argv[2]);
  imu.host_imu_data_port = 56401;
  imu.lidar_imu_data_port = 56400;

  HostStateInfoIpInfo state{};
  std::snprintf(state.host_ip_addr, sizeof(state.host_ip_addr), "%s", argv[2]);
  state.host_state_info_port = 56201;
  state.lidar_state_info_port = 56200;

  const bool point_ok = SetAndWait(
      "point cloud -> Unitree", point, SetLivoxLidarPointDataHostIPCfg);
  const bool imu_ok = SetAndWait(
      "IMU -> Unitree", imu, SetLivoxLidarImuDataHostIPCfg);
  const bool state_ok = SetAndWait(
      "state info -> Unitree", state, SetLivoxLidarStateInfoHostIPCfg);

  LivoxLidarSdkUninit();
  if (!(pcl_ok && pattern_ok && normal_ok && imu_send_ok &&
        point_ok && imu_ok && state_ok)) {
    return 5;
  }
  std::printf("[mid360-restore] destinațiile native au fost restaurate la %s\n",
              argv[2]);
  return 0;
}
