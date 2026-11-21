"""Tests for the BPIO2 hardware abstraction layer."""

import io
import sys

import pytest
from unittest.mock import patch, MagicMock
from buspirate_mcp.hardware import BusPirateHardware


class TestListDevices:
    def test_finds_acm_devices(self, tmp_path):
        with patch("buspirate_mcp.hardware.glob") as mock_glob:
            mock_glob.return_value = ["/dev/ttyACM0", "/dev/ttyACM1"]
            devices = BusPirateHardware.list_devices()
            assert len(devices) == 2
            assert devices[0]["path"] == "/dev/ttyACM0"
            assert devices[1]["path"] == "/dev/ttyACM1"

    def test_identifies_terminal_and_binary_ports(self):
        with patch("buspirate_mcp.hardware.glob") as mock_glob:
            mock_glob.return_value = ["/dev/ttyACM0", "/dev/ttyACM1"]
            devices = BusPirateHardware.list_devices()
            assert devices[0]["role"] == "terminal"
            assert devices[1]["role"] == "binary"

    def test_single_device_marked_unknown(self):
        with patch("buspirate_mcp.hardware.glob") as mock_glob:
            mock_glob.return_value = ["/dev/ttyACM0"]
            devices = BusPirateHardware.list_devices()
            assert devices[0]["role"] == "unknown"

    def test_returns_empty_when_no_devices(self):
        with patch("buspirate_mcp.hardware.glob") as mock_glob:
            mock_glob.return_value = []
            devices = BusPirateHardware.list_devices()
            assert devices == []


class TestConnect:
    def test_connect_creates_client_and_uart(self):
        with patch("buspirate_mcp.hardware.BPIOClient") as mock_client_cls, \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_client = MagicMock()
            mock_client.status_request.return_value = {
                "mode_current": "HiZ",
                "version_hardware_major": 6,
            }
            mock_client_cls.return_value = mock_client
            mock_uart_cls.return_value = MagicMock()

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            assert hw.client is not None
            assert hw.uart is not None
            mock_client_cls.assert_called_once_with("/dev/ttyACM1")

    def test_connect_raises_on_failure(self):
        with patch("buspirate_mcp.hardware.BPIOClient") as mock_client_cls:
            mock_client_cls.side_effect = Exception("USB not found")
            with pytest.raises(ConnectionError, match="USB not found"):
                BusPirateHardware.connect("/dev/ttyACM99")


class TestUARTOperations:
    def test_configure_uart(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            hw.configure_uart(speed=9600)
            mock_uart.configure.assert_called_once_with(
                speed=9600, data_bits=8, parity=False,
                stop_bits=1, flow_control=False,
                signal_inversion=False, async_callback=None,
            )

    def test_read_returns_bytes(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart.read_async.return_value = b"Hello\n"
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            data = hw.read()
            assert data == b"Hello\n"

    def test_write_sends_bytes(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            hw.write(b"AT\r\n")
            mock_uart.transfer.assert_called_once_with(b"AT\r\n", read_bytes=0)


class TestPowerSupply:
    """PSU requests go straight to the client: they need no configured mode."""

    def _hw(self):
        client = MagicMock()
        client.configuration_request.return_value = True
        return BusPirateHardware(client=client, uart=MagicMock()), client

    def test_set_voltage(self):
        hw, client = self._hw()
        assert hw.set_voltage(3.3, 300) is True
        client.configuration_request.assert_called_once_with(
            psu_enable=True, psu_set_mv=3300, psu_set_ma=300,
        )

    def test_set_voltage_uses_round(self):
        hw, client = self._hw()
        hw.set_voltage(2.3, 100)
        # round(2.3 * 1000) = 2300, not int(2.3 * 1000) = 2299
        assert client.configuration_request.call_args.kwargs["psu_set_mv"] == 2300

    def test_set_power_on_uses_last_voltage(self):
        hw, client = self._hw()
        hw.set_voltage(1.8, 100)
        hw.set_power(False)
        hw.set_power(True)
        # Should re-apply 1.8V, not SDK default 3.3V
        calls = client.configuration_request.call_args_list
        assert calls[1].kwargs == {"psu_disable": True}
        assert calls[-1].kwargs == {
            "psu_enable": True, "psu_set_mv": 1800, "psu_set_ma": 100,
        }

    def test_set_power_on_without_voltage_raises(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            with pytest.raises(RuntimeError, match="No voltage configured"):
                hw.set_power(True)

    def test_get_pin_voltages(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart.get_adc_mv.return_value = [0, 0, 0, 0, 3300, 3280, 0, 0]
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            voltages = hw.get_pin_voltages()
            assert voltages == [0, 0, 0, 0, 3300, 3280, 0, 0]


class TestDisconnect:
    def test_disconnect_disables_psu(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            client = hw.client
            hw.disconnect()
            client.configuration_request.assert_called_once_with(psu_disable=True)
            assert hw.client is None
            assert hw.uart is None

    def test_disconnect_safe_when_uart_gone(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart.set_psu_disable.side_effect = Exception("USB gone")
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            hw.disconnect()  # should not raise
            assert hw.uart is None

    def test_disconnect_nulls_all_protocols(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart

            hw = BusPirateHardware.connect("/dev/ttyACM1")
            hw.spi = MagicMock()
            hw.i2c = MagicMock()
            hw.onewire = MagicMock()
            hw.disconnect()
            assert hw.spi is None
            assert hw.i2c is None
            assert hw.onewire is None
            assert hw._active_mode is None


class TestModeTracking:
    def _make_hw(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart
            return BusPirateHardware.connect("/dev/ttyACM1")

    def test_initial_mode_is_none(self):
        hw = self._make_hw()
        assert hw._active_mode is None

    def test_configure_uart_sets_mode(self):
        hw = self._make_hw()
        hw.configure_uart()
        assert hw._active_mode == "uart"

    def test_check_mode_raises_on_switch(self):
        hw = self._make_hw()
        hw._active_mode = "uart"
        with pytest.raises(RuntimeError, match="Mode 'uart' is active"):
            hw._check_mode("spi")

    def test_check_mode_allows_same(self):
        hw = self._make_hw()
        hw._active_mode = "spi"
        hw._check_mode("spi")  # should not raise

    def test_reset_mode_allows_switch(self):
        hw = self._make_hw()
        hw._active_mode = "uart"
        hw.reset_mode()
        assert hw._active_mode is None
        hw._check_mode("spi")  # should not raise

    def test_active_protocol_returns_uart_by_default(self):
        hw = self._make_hw()
        assert hw._active_protocol is hw.uart

    def test_active_protocol_returns_spi(self):
        hw = self._make_hw()
        hw.spi = MagicMock()
        hw._active_mode = "spi"
        assert hw._active_protocol is hw.spi

    def test_active_protocol_returns_i2c(self):
        hw = self._make_hw()
        hw.i2c = MagicMock()
        hw._active_mode = "i2c"
        assert hw._active_protocol is hw.i2c

    def test_active_protocol_returns_onewire(self):
        hw = self._make_hw()
        hw.onewire = MagicMock()
        hw._active_mode = "1wire"
        assert hw._active_protocol is hw.onewire

    def test_psu_does_not_depend_on_active_protocol(self):
        hw = self._make_hw()
        hw.client.configuration_request.return_value = True
        mock_spi = MagicMock()
        hw.spi = mock_spi
        hw._active_mode = "spi"
        hw.set_voltage(3.3, 300)
        mock_spi.set_psu_enable.assert_not_called()
        hw.client.configuration_request.assert_called_once_with(
            psu_enable=True, psu_set_mv=3300, psu_set_ma=300,
        )


class TestSPIOperations:
    def _make_hw(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart
            return BusPirateHardware.connect("/dev/ttyACM1")

    def test_configure_spi_sets_mode(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIOSPI") as mock_spi_cls:
            mock_spi = MagicMock()
            mock_spi_cls.return_value = mock_spi
            hw.configure_spi(speed=2000000)
            assert hw._active_mode == "spi"
            mock_spi.configure.assert_called_once_with(
                speed=2000000, clock_polarity=False,
                clock_phase=False, chip_select_idle=True,
            )

    def test_configure_spi_with_psu(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIOSPI") as mock_spi_cls:
            mock_spi = MagicMock()
            mock_spi_cls.return_value = mock_spi
            hw.configure_spi(voltage_mv=3300, current_ma=200)
            mock_spi.configure.assert_called_once_with(
                speed=1000000, clock_polarity=False,
                clock_phase=False, chip_select_idle=True,
                psu_enable=True, psu_set_mv=3300, psu_set_ma=200,
            )

    def test_configure_spi_reuses_existing_object(self):
        hw = self._make_hw()
        mock_spi = MagicMock()
        hw.spi = mock_spi
        hw.configure_spi()
        assert hw.spi is mock_spi

    def test_spi_transfer(self):
        hw = self._make_hw()
        mock_spi = MagicMock()
        mock_spi.transfer.return_value = b"\xff"
        hw.spi = mock_spi
        hw._active_mode = "spi"
        result = hw.spi_transfer(b"\x00", read_bytes=1)
        mock_spi.transfer.assert_called_once_with(
            write_data=b"\x00", read_bytes=1,
        )
        assert result == b"\xff"

    def test_spi_select_deselect(self):
        hw = self._make_hw()
        mock_spi = MagicMock()
        hw.spi = mock_spi
        hw._active_mode = "spi"
        hw.spi_select()
        mock_spi.select.assert_called_once()
        hw.spi_deselect()
        mock_spi.deselect.assert_called_once()

    def test_configure_spi_blocked_by_uart_mode(self):
        hw = self._make_hw()
        hw.configure_uart()
        with pytest.raises(RuntimeError, match="Mode 'uart' is active"):
            hw.configure_spi()


class TestI2COperations:
    def _make_hw(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart
            return BusPirateHardware.connect("/dev/ttyACM1")

    def test_configure_i2c_sets_mode_with_pullups(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIOI2C") as mock_i2c_cls:
            mock_i2c = MagicMock()
            mock_i2c_cls.return_value = mock_i2c
            hw.configure_i2c(speed=100000)
            assert hw._active_mode == "i2c"
            mock_i2c.configure.assert_called_once_with(
                speed=100000, clock_stretch=False,
                pullup_enable=True,
            )

    def test_configure_i2c_with_psu(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIOI2C") as mock_i2c_cls:
            mock_i2c = MagicMock()
            mock_i2c_cls.return_value = mock_i2c
            hw.configure_i2c(voltage_mv=3300, current_ma=150)
            mock_i2c.configure.assert_called_once_with(
                speed=400000, clock_stretch=False,
                pullup_enable=True,
                psu_enable=True, psu_set_mv=3300, psu_set_ma=150,
            )

    def test_i2c_transfer(self):
        hw = self._make_hw()
        mock_i2c = MagicMock()
        mock_i2c.transfer.return_value = b"\xab"
        hw.i2c = mock_i2c
        hw._active_mode = "i2c"
        result = hw.i2c_transfer(b"\x50", read_bytes=1)
        mock_i2c.transfer.assert_called_once_with(
            write_data=b"\x50", read_bytes=1,
        )
        assert result == b"\xab"

    def test_i2c_scan_suppresses_stdout(self):
        hw = self._make_hw()
        mock_i2c = MagicMock()

        def noisy_scan(start_addr=0, end_addr=0x7F):
            print("Scanning...")  # this must not reach real stdout
            return [0x48, 0x50]

        mock_i2c.scan = noisy_scan
        hw.i2c = mock_i2c
        hw._active_mode = "i2c"

        # Capture real stdout to verify nothing leaks
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            result = hw.i2c_scan()
        finally:
            sys.stdout = old_stdout

        assert result == [0x48, 0x50]
        assert captured.getvalue() == ""


class TestOneWireOperations:
    def _make_hw(self):
        with patch("buspirate_mcp.hardware.BPIOClient"), \
             patch("buspirate_mcp.hardware.BPIOUART") as mock_uart_cls:
            mock_uart = MagicMock()
            mock_uart_cls.return_value = mock_uart
            return BusPirateHardware.connect("/dev/ttyACM1")

    def test_configure_1wire_sets_mode(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIO1Wire") as mock_ow_cls:
            mock_ow = MagicMock()
            mock_ow_cls.return_value = mock_ow
            hw.configure_1wire()
            assert hw._active_mode == "1wire"
            mock_ow.configure.assert_called_once_with(pullup_enable=True)

    def test_configure_1wire_with_psu(self):
        hw = self._make_hw()
        with patch("buspirate_mcp.hardware.BPIO1Wire") as mock_ow_cls:
            mock_ow = MagicMock()
            mock_ow_cls.return_value = mock_ow
            hw.configure_1wire(voltage_mv=3300, current_ma=50)
            mock_ow.configure.assert_called_once_with(
                pullup_enable=True,
                psu_enable=True, psu_set_mv=3300, psu_set_ma=50,
            )

    def test_onewire_reset(self):
        hw = self._make_hw()
        mock_ow = MagicMock()
        mock_ow.reset.return_value = True
        hw.onewire = mock_ow
        hw._active_mode = "1wire"
        result = hw.onewire_reset()
        mock_ow.reset.assert_called_once()
        assert result is True

    def test_onewire_transfer(self):
        hw = self._make_hw()
        mock_ow = MagicMock()
        mock_ow.transfer.return_value = b"\x28"
        hw.onewire = mock_ow
        hw._active_mode = "1wire"
        result = hw.onewire_transfer(b"\x33", read_bytes=1)
        mock_ow.transfer.assert_called_once_with(
            write_data=b"\x33", read_bytes=1,
        )
        assert result == b"\x28"

    def test_configure_1wire_blocked_by_spi_mode(self):
        hw = self._make_hw()
        hw._active_mode = "spi"
        with pytest.raises(RuntimeError, match="Mode 'spi' is active"):
            hw.configure_1wire()


def _config_response(error=None):
    """Raw BPIO2 ConfigurationResponse bytes, as the firmware sends them."""
    import flatbuffers
    from tooling.bpio import (
        ConfigurationResponse, ResponsePacket, ResponsePacketContents,
    )
    b = flatbuffers.Builder(64)
    err = b.CreateString(error) if error else None
    ConfigurationResponse.Start(b)
    if err:
        ConfigurationResponse.AddError(b, err)
    contents = ConfigurationResponse.End(b)
    ResponsePacket.Start(b)
    ResponsePacket.AddContentsType(
        b, ResponsePacketContents.ResponsePacketContents.ConfigurationResponse,
    )
    ResponsePacket.AddContents(b, contents)
    b.Finish(ResponsePacket.End(b))
    return bytes(b.Output())


class TestErrorPropagation:
    """Failures raise BPIOError with the firmware's reason, not False/None."""

    def _hw(self, response):
        client = MagicMock()

        def send_and_receive(data):
            return response

        client.send_and_receive = send_and_receive

        def configuration_request(**kwargs):
            # what the SDK does: send, then collapse any error to False
            resp = client.send_and_receive(b"req")
            return bool(resp) and b"out of range" not in resp

        client.configuration_request.side_effect = configuration_request
        return BusPirateHardware(client=client, uart=MagicMock())

    def test_firmware_error_string_is_raised(self):
        hw = self._hw(_config_response("PSU Voltage out of range (800-5000 mV)"))
        from buspirate_mcp.hardware import BPIOError
        with pytest.raises(BPIOError, match="set voltage: PSU Voltage out of range"):
            hw.set_voltage(0.5, 100)

    def test_no_response_is_reported_as_timeout(self):
        hw = self._hw(False)
        from buspirate_mcp.hardware import BPIOError
        with pytest.raises(BPIOError, match="no response from BusPirate"):
            hw.set_power(False)

    def test_failed_transfer_raises_instead_of_returning_false(self):
        hw = self._hw(None)
        hw.i2c = MagicMock()
        hw.i2c.transfer.return_value = False
        from buspirate_mcp.hardware import BPIOError
        with pytest.raises(BPIOError, match="I2C transfer"):
            hw.i2c_transfer([0xA0], 1)

    def test_transfer_without_read_data_returns_empty_bytes(self):
        hw = self._hw(None)
        hw.spi = MagicMock()
        hw.spi.transfer.return_value = None
        assert hw.spi_transfer([0x06], 0) == b""

    def test_onewire_reset_without_presence_is_false(self):
        hw = self._hw(_config_response())  # a response arrived
        hw.onewire = MagicMock()
        hw.onewire.reset.return_value = False
        hw.client._bpio_last_response = b"something"
        assert hw.onewire_reset() is False

    def test_configure_failure_raises(self):
        hw = self._hw(None)
        hw.uart.configure.return_value = False
        from buspirate_mcp.hardware import BPIOError
        with pytest.raises(BPIOError, match="configure UART"):
            hw.configure_uart(115200)
