# utils/packet_utils.py

import math
import decimal
from collections import Counter
from scapy.layers.inet import IP
from scapy.data import IP_PROTOS

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


def drop_columns(df, columns):
    """
    Drops specified columns from a DataFrame if they exist.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to drop

    Returns:
    - Modified DataFrame
    """
    for col in columns:
        if col in df.columns:
            logger.debug("Dropping column: %s", col)

    return df.drop(columns=[c for c in columns if c in df.columns], errors='ignore')


def print_packet_summary(pkt):
    if IP in pkt:
        pkt_src = pkt[IP].src
        pkt_dst = pkt[IP].dst
        proto_num = int(pkt[IP].proto)
        proto_lookup = dict(IP_PROTOS)
        pkt_proto = proto_lookup.get(proto_num, f"IP#{proto_num}")
    else:
        pkt_src = getattr(pkt, 'src', 'unknown')
        pkt_dst = getattr(pkt, 'dst', 'unknown')
        pkt_proto = pkt.name

    logger.info("Packet: %s → %s | Protocol: %s | Length: %s",
                pkt_src, pkt_dst, pkt_proto, len(pkt))


def calc_entropy(data: bytes):
    if not data:
        return 0.0

    length = len(data)
    counts = Counter(data)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())

    return round(entropy, 4)


def safe_numeric_cast(x):
    if isinstance(x, (float, int)):
        return x
    if isinstance(x, decimal.Decimal):
        return float(x)
    if hasattr(x, '__float__'):
        try:
            return float(x)
        except:
            return x
    return x



