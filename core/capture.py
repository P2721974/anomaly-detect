# core/capture.py

from tqdm import tqdm
from scapy.all import sniff, Scapy_Exception

from utils.packet_utils import print_packet_summary
from utils.file_saver import safe_save_path, save_pcap
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def capture_packets(interface, duration, packet_count, output_path):
    logger.info(f"Capturing packets on interface '%s' for %ds or %d packets...", interface, duration, packet_count)
    try:
        captured = []
        bar = tqdm(total=packet_count, desc="Capturing Packets", unit="pkt", leave=True)
        
        # defined within a function again -> change if poss.
        def handle(pkt):
            captured.append(pkt)
            bar.update(1)

        sniff(
            iface=interface, 
            timeout=duration, 
            count=packet_count,
            prn=handle,
            )
        
        bar.close()

        logger.info("Capture complete.")

        # Save
        pcap_path = safe_save_path(output_path)
        save_pcap(captured, pcap_path)

    except Scapy_Exception as e:
        logger.error("Packet capture failed: %s", e)
    except Exception as e:
        logger.exception("Unexpected error during capture: %s", e)


def run_capture(args):
    interface = args.interface or config['capture']['interface']
    duration = args.duration or config['capture']['duration']
    packet_count = args.packet_count or config['capture']['packet_count']
    output_path = args.output or config['capture']['output_path']

    capture_packets(interface, duration, packet_count, output_path)


def live_packet_monitor(interface, packet_callback, count, timeout):
    """
    Sniff packets from a live interface and forward them to a callback.

    Parameters:
    - interface: Interface to listen on
    - packet_callback: Function to call for each packet
    - count: Optional number of packets to stop capture after
    - timeout: Optional timeout in seconds
    """

    logger.info("Starting live capture on interface: %s", interface)
    sniff(iface=interface, prn=packet_callback, store=False, count=count, timeout=timeout)


def run_live_capture(args):
    interface = args.interface or config['capture']['interface']
    count = args.packet_count or 0
    duration = args.duration or None

    live_packet_monitor(interface, print_packet_summary, count, duration)


