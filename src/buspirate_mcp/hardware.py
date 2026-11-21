"""Thin abstraction over the BPIO2 SDK.

Only this module imports from the BPIO2 SDK (bpio_client, bpio_uart, etc.).
Everything else in the codebase talks to this module. This makes
testing without hardware trivial -- mock this, not the SDK.
"""

from __future__ import annotations

import io
import sys
from glob import glob
from pathlib import Path
from typing import Any

# Add vendored BPIO2 SDK to sys.path so we can import it.
# The SDK lives at vendor/bpio2/python/pybpio/ relative to the package root.
_VENDOR_PATH = str(
    Path(__file__).resolve().parents[2] / "vendor" / "bpio2" / "python"
)
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

try:
    from pybpio.bpio_client import BPIOClient
    from pybpio.bpio_uart import BPIOUART
    from pybpio.bpio_spi import BPIOSPI
    from pybpio.bpio_i2c import BPIOI2C
    from pybpio.bpio_1wire import BPIO1Wire
    from tooling.bpio import (
        ConfigurationResponse,
        DataResponse,
        ResponsePacket,
        ResponsePacketContents,
    )
except ImportError:
    # SDK not available -- tests use mocks, but fail fast at runtime
    BPIOClient = None  # type: ignore[assignment, misc]
    BPIOUART = None  # type: ignore[assignment, misc]
    BPIOSPI = None  # type: ignore[assignment, misc]
    BPIOI2C = None  # type: ignore[assignment, misc]
    BPIO1Wire = None  # type: ignore[assignment, misc]
    ResponsePacket = None  # type: ignore[assignment]


class BPIOError(RuntimeError):
    """A BPIO2 request failed. The message carries the firmware's reason."""


def _capture_responses(client: Any) -> None:
    """Keep the last raw response on the client so failures can be explained.

    The SDK turns every failure into False/None and only print()s the reason,
    so wrap send_and_receive and decode the error ourselves.
    """
    send = getattr(client, "send_and_receive", None)
    if send is None or getattr(send, "_bpio_capture", False):
        return

    def send_and_capture(data: Any) -> Any:
        client._bpio_last_response = None
        resp = send(data)
        client._bpio_last_response = resp
        return resp

    send_and_capture._bpio_capture = True  # type: ignore[attr-defined]
    client.send_and_receive = send_and_capture


def _response_error(resp: Any) -> str | None:
    """Error string from a raw BPIO2 response, wherever the firmware put it."""
    if not resp or ResponsePacket is None:
        return None
    try:
        packet = ResponsePacket.ResponsePacket.GetRootAsResponsePacket(resp, 0)
        if packet.Error():
            return packet.Error().decode("utf-8", errors="replace")
        kinds = ResponsePacketContents.ResponsePacketContents
        table = {
            kinds.ConfigurationResponse: ConfigurationResponse.ConfigurationResponse,
            kinds.DataResponse: DataResponse.DataResponse,
        }.get(packet.ContentsType())
        if table is None or packet.Contents() is None:
            return None
        contents = table()
        contents.Init(packet.Contents().Bytes, packet.Contents().Pos)
        err = contents.Error()
        return err.decode("utf-8", errors="replace") if err else None
    except Exception:
        return None


# pid.codes VID:PID for Bus Pirate 5/6/7 (CDC) and its logic analyzer USB config
BUSPIRATE_USB_IDS = {(0x1209, 0x7331), (0x1209, 0x7332)}
_INTERFACE_ROLES = {0: "terminal", 2: "binary"}


def _interface_number(location: str | None) -> int | None:
    """USB interface number from a pyserial location like '1-2.3:1.2'."""
    if not location or ":" not in location:
        return None
    try:
        return int(location.rsplit(".", 1)[-1])
    except ValueError:
        return None


def _usb_ports() -> list[dict[str, str]]:
    """Bus Pirate serial ports found by USB VID:PID, sorted by path."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    devices = []
    for port in list_ports.comports():
        if (port.vid, port.pid) not in BUSPIRATE_USB_IDS:
            continue
        role = _INTERFACE_ROLES.get(_interface_number(port.location), "unknown")
        device = {"path": port.device, "role": role}
        if port.serial_number:
            device["serial_number"] = port.serial_number
        devices.append(device)
    return sorted(devices, key=lambda d: d["path"])


class BusPirateHardware:
    """Wraps a single BusPirate 6 connection."""

    def __init__(self, client: Any, uart: Any) -> None:
        self.client = client
        self.uart = uart
        self.spi: Any = None
        self.i2c: Any = None
        self.onewire: Any = None
        self._active_mode: str | None = None
        self._last_voltage_mv: int = 0
        self._last_current_ma: int = 0
        _capture_responses(client)

    # -- error handling --

    def _failure(self, what: str) -> BPIOError:
        resp = getattr(self.client, "_bpio_last_response", None)
        if not resp:
            return BPIOError(f"{what}: no response from BusPirate (timeout)")
        reason = _response_error(resp) or "request failed"
        return BPIOError(f"{what}: {reason}")

    def _config(self, what: str, **kwargs: Any) -> bool:
        """Send a ConfigurationRequest; raise BPIOError with the reason on failure.

        Goes to the client directly: the SDK protocol objects refuse
        (print + None) until their mode is configured, which made PSU and
        GPIO calls silently do nothing before open_uart/open_spi/etc.
        """
        if self.client.configuration_request(**kwargs) is not True:
            raise self._failure(what)
        return True

    def _data(self, what: str, result: Any) -> bytes:
        """Check a data request result: False is a failure, None is no data."""
        if result is False:
            raise self._failure(what)
        return bytes(result) if result else b""

    def _configured(self, what: str, ok: Any) -> None:
        if not ok:
            raise self._failure(what)

    @property
    def _active_protocol(self) -> Any:
        """Return the protocol object for the currently active mode.

        Falls back to self.uart when no mode is set (backward compat for
        PSU calls before any protocol is configured).
        """
        if self._active_mode == "spi":
            return self.spi
        if self._active_mode == "i2c":
            return self.i2c
        if self._active_mode == "1wire":
            return self.onewire
        # "uart" or None -- use uart as default
        return self.uart

    def _check_mode(self, target_mode: str) -> None:
        """Raise if a different mode is already active.

        Reconfiguring the same mode is allowed. Switching requires
        reset_mode() first.
        """
        if self._active_mode is not None and self._active_mode != target_mode:
            raise RuntimeError(
                f"Mode '{self._active_mode}' is active. "
                f"Call reset_mode() before switching to '{target_mode}'."
            )

    def reset_mode(self) -> None:
        """Clear the active mode so a different protocol can be configured."""
        self._active_mode = None

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        """Find BusPirate devices on USB CDC serial ports.

        BP5/6/7 expose two CDC ports: interface 0 is the text terminal,
        interface 2 is the binary protocol (BPIO2/FALA). Ports are matched by
        USB VID:PID and the role comes from the interface number, so it stays
        right when ACM numbers shift after a replug or another CDC device is
        attached. Falls back to guessing from /dev/ttyACM* order when pyserial
        reports no Bus Pirate (e.g. no USB metadata available).
        """
        usb = _usb_ports()
        if usb:
            return usb
        paths = sorted(glob("/dev/ttyACM*"))
        devices = []
        for i, p in enumerate(paths):
            if len(paths) >= 2:
                role = "terminal" if i % 2 == 0 else "binary"
            else:
                role = "unknown"
            devices.append({"path": p, "role": role})
        return devices

    @classmethod
    def connect(cls, port: str) -> BusPirateHardware:
        """Open a connection to a BusPirate 6 on the given serial port."""
        if BPIOClient is None:
            raise ImportError(
                "BPIO2 SDK not installed. "
                "See https://docs.buspirate.com/docs/binmode-reference/protocol-bpio2/"
            )
        try:
            client = BPIOClient(port)
            uart = BPIOUART(client)
            return cls(client=client, uart=uart)
        except Exception as exc:
            raise ConnectionError(str(exc)) from exc

    def configure_uart(
        self,
        speed: int = 115200,
        data_bits: int = 8,
        parity: bool = False,
        stop_bits: int = 1,
    ) -> None:
        """Configure UART mode on the BusPirate."""
        self._check_mode("uart")
        ok = self.uart.configure(
            speed=speed,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            flow_control=False,
            signal_inversion=False,
            async_callback=None,
        )
        self._configured("configure UART", ok)
        self._active_mode = "uart"

    def read(self) -> bytes:
        """Read whatever is in the UART receive buffer."""
        return self.uart.read_async()

    def write(self, data: bytes) -> None:
        """Send data over UART to the target."""
        self.uart.transfer(data, read_bytes=0)

    def set_voltage(self, voltage_v: float, current_limit_ma: int) -> bool:
        """Set the power supply voltage and current limit."""
        voltage_mv = round(voltage_v * 1000)
        self._last_voltage_mv = voltage_mv
        self._last_current_ma = current_limit_ma
        return self._config(
            "set voltage", psu_enable=True,
            psu_set_mv=voltage_mv, psu_set_ma=current_limit_ma,
        )

    def set_power(self, enable: bool) -> bool:
        """Enable or disable the power supply.

        When enabling, re-applies the last configured voltage/current.
        Raises RuntimeError if no voltage was previously configured.
        """
        if enable:
            if self._last_voltage_mv == 0:
                raise RuntimeError(
                    "No voltage configured. Call set_voltage() before set_power(True)."
                )
            return self._config(
                "power on", psu_enable=True,
                psu_set_mv=self._last_voltage_mv,
                psu_set_ma=self._last_current_ma,
            )
        return self._config("power off", psu_disable=True)

    @staticmethod
    def _validate_pin(pin: int) -> None:
        """Raise ValueError if pin is not a valid IO pin number (0-7)."""
        if not isinstance(pin, int) or pin < 0 or pin > 7:
            raise ValueError(f"Pin must be 0-7, got: {pin}")

    def configure_pin_input(self, pin: int) -> None:
        """Configure a pin as digital input for reading."""
        self._validate_pin(pin)
        mask = 1 << pin
        self._config(f"IO{pin} input", io_direction_mask=mask, io_direction=0)

    def get_status(self) -> dict[str, Any]:
        """Full BPIO2 status: versions, mode, PSU readings, pins.

        Goes to the client directly so it works before any mode is set.
        """
        status = self.client.status_request()
        if not status:
            raise RuntimeError("status request failed: no response from BusPirate")
        return status

    def set_pullups(self, enable: bool) -> bool:
        """Enable or disable the on-board pull-up resistors (to VOUT)."""
        key = "pullup_enable" if enable else "pullup_disable"
        if self.client.configuration_request(**{key: True}) is not True:
            raise RuntimeError(f"pull-up {'enable' if enable else 'disable'} failed")
        return True

    def get_pin_voltages(self) -> list[int]:
        """Read ADC millivolt values for all IO pins."""
        return self._active_protocol.get_adc_mv()

    def set_pin_output(self, pin: int, high: bool) -> None:
        """Set a pin as output and drive it high or low."""
        self._validate_pin(pin)
        mask = 1 << pin
        # value first, then direction, so the pin never drives the old level
        self._config(
            f"IO{pin} output", io_value_mask=mask, io_value=mask if high else 0,
        )
        self._config(f"IO{pin} output", io_direction_mask=mask, io_direction=mask)

    def release_pin(self, pin: int) -> None:
        """Release a pin back to input (high-impedance)."""
        self._validate_pin(pin)
        mask = 1 << pin
        self._config(f"IO{pin} input", io_direction_mask=mask, io_direction=0)

    # -- SPI --

    def configure_spi(
        self,
        speed: int = 1000000,
        clock_polarity: bool = False,
        clock_phase: bool = False,
        chip_select_idle: bool = True,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure SPI mode on the BusPirate."""
        self._check_mode("spi")
        if self.spi is None:
            self.spi = BPIOSPI(self.client)
        kwargs: dict[str, Any] = {}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        ok = self.spi.configure(
            speed=speed,
            clock_polarity=clock_polarity,
            clock_phase=clock_phase,
            chip_select_idle=chip_select_idle,
            **kwargs,
        )
        self._configured("configure SPI", ok)
        self._active_mode = "spi"

    def spi_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over SPI."""
        return self._data(
            "SPI transfer",
            self.spi.transfer(write_data=write_data, read_bytes=read_bytes),
        )

    def spi_select(self) -> None:
        """Assert chip select (active low)."""
        self._data("SPI select", self.spi.select())

    def spi_deselect(self) -> None:
        """Deassert chip select."""
        self._data("SPI deselect", self.spi.deselect())

    # -- I2C --

    def configure_i2c(
        self,
        speed: int = 400000,
        clock_stretch: bool = False,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure I2C mode on the BusPirate."""
        self._check_mode("i2c")
        if self.i2c is None:
            self.i2c = BPIOI2C(self.client)
        kwargs: dict[str, Any] = {"pullup_enable": True}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        ok = self.i2c.configure(speed=speed, clock_stretch=clock_stretch, **kwargs)
        self._configured("configure I2C", ok)
        self._active_mode = "i2c"

    def i2c_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over I2C."""
        return self._data(
            "I2C transfer",
            self.i2c.transfer(write_data=write_data, read_bytes=read_bytes),
        )

    def i2c_scan(self, start_addr: int = 0x00, end_addr: int = 0x7F) -> list:
        """Scan the I2C bus for devices.

        The SDK scan() has print() calls that would corrupt MCP stdio
        transport, so we redirect stdout during the call.
        """
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return self.i2c.scan(start_addr=start_addr, end_addr=end_addr)
        finally:
            sys.stdout = old_stdout

    # -- 1-Wire --

    def configure_1wire(
        self,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure 1-Wire mode on the BusPirate."""
        self._check_mode("1wire")
        if self.onewire is None:
            self.onewire = BPIO1Wire(self.client)
        kwargs: dict[str, Any] = {"pullup_enable": True}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        ok = self.onewire.configure(**kwargs)
        self._configured("configure 1-Wire", ok)
        self._active_mode = "1wire"

    def onewire_reset(self) -> Any:
        """Send a 1-Wire bus reset pulse. True if a device answered.

        The firmware reports "no presence pulse" as a failed request, so a
        failure with a response means no device; no response at all is an
        error.
        """
        if self.onewire.reset() is not False:
            return True
        if not getattr(self.client, "_bpio_last_response", None):
            raise self._failure("1-Wire reset")
        return False

    def onewire_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over 1-Wire."""
        return self._data(
            "1-Wire transfer",
            self.onewire.transfer(write_data=write_data, read_bytes=read_bytes),
        )

    @staticmethod
    def find_terminal_port() -> str | None:
        """Find the BP6 terminal port (first ACM device)."""
        devices = BusPirateHardware.list_devices()
        for d in devices:
            if d["role"] == "terminal":
                return d["path"]
        return devices[0]["path"] if devices else None

    def disconnect(self) -> None:
        """Disable PSU and clean up the connection."""
        if self.client is not None:
            try:
                self.client.configuration_request(psu_disable=True)
            except Exception:
                pass  # best effort -- hardware may already be gone
        self.client = None
        self.uart = None
        self.spi = None
        self.i2c = None
        self.onewire = None
        self._active_mode = None
