# core/preprocessor.py

import numpy as np
import pandas as pd
from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.packet import Raw

from utils.packet_utils import drop_columns, calc_entropy, safe_numeric_cast
from utils.file_saver import safe_save_path, save_dataframe
from utils.progress import tqdm_bar
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def extract_packet_fields(pkt, last_pkt_time=None):
    row = {
        # General
        'timestamp': float(pkt.time),
        'length': len(pkt),
        'inter_arrival_time': (pkt.time - last_pkt_time) if last_pkt_time else 0.0,

        # IP
        'ip_proto_number': None,
        'ip_ttl': None,
        'ip_id': None,
        'ip_flags_df': 0,
        'ip_flags_mf': 0,
        'wrong_fragment': 0,

        # TCP
        'tcp_sport': None,
        'tcp_dport': None,
        'tcp_seq': None,
        'tcp_ack': None,
        'tcp_flag_int': 0,
        'tcp_flag_syn': 0,
        'tcp_flag_ack': 0,
        'tcp_flag_fin': 0,
        'tcp_flag_rst': 0,
        'tcp_flag_psh': 0,
        'tcp_flag_urg': 0,
        'urgent_pointer': 0,
        'tcp_window': None,
        'tcp_dataofs': None,

        # UDP
        'udp_sport': None,
        'udp_dport': None,
        'udp_len': None,

        # ICMP
        'icmp_type': None,
        'icmp_code': None,

        # Payload
        'payload_size': 0,
        'is_empty_payload': 1,
        'payload_entropy': 0.0,
    }

    try:
        if IP in pkt:
            ip_layer = pkt[IP]
            row['ip_proto_number'] = int(ip_layer.proto)
            row['ip_ttl'] = int(ip_layer.ttl)
            row['ip_id'] = int(ip_layer.id)
            row['ip_flags_df'] = int(ip_layer.flags.DF)
            row['ip_flags_mf'] = int(ip_layer.flags.MF)
            row['wrong_fragment'] = 1 if ip_layer.frag > 0 else 0

        if TCP in pkt:
            tcp = pkt[TCP]
            row['tcp_sport'] = tcp.sport
            row['tcp_dport'] = tcp.dport
            row['tcp_seq'] = tcp.seq
            row['tcp_ack'] = tcp.ack
            row['tcp_flag_int'] = int(tcp.flags)
            row['tcp_flag_syn'] = int(tcp.flags.S)
            row['tcp_flag_ack'] = int(tcp.flags.A)
            row['tcp_flag_fin'] = int(tcp.flags.F)
            row['tcp_flag_rst'] = int(tcp.flags.R)
            row['tcp_flag_psh'] = int(tcp.flags.P)
            row['tcp_flag_urg'] = int(tcp.flags.U)
            row['urgent_pointer'] = int(pkt[TCP].urgptr) if hasattr(pkt[TCP], 'urgptr') else 0
            row['tcp_window'] = tcp.window
            row['tcp_dataofs'] = tcp.dataofs

        elif UDP in pkt:
            udp = pkt[UDP]
            row['udp_sport'] = udp.sport
            row['udp_dport'] = udp.dport
            row['udp_len'] = udp.len

        elif ICMP in pkt:
            icmp = pkt[ICMP]
            row['icmp_type'] = icmp.type
            row['icmp_code'] = icmp.code

        if Raw in pkt:
            payload = bytes(pkt[Raw])
            row['payload_size'] = len(payload)
            row['is_empty_payload'] = int(len(payload) == 0)
            row['payload_entropy'] = calc_entropy(payload)

    except Exception as e:
        logger.warning("Packet parsing error: %s", e)

    return row


def extract_packet_features(pkt):
    row = extract_packet_fields(pkt)
    return pd.DataFrame([row]) if row else pd.DataFrame()


def clean_dataframe(df):
    # Replace infs with NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN in all numeric features
    df = df.applymap(safe_numeric_cast)     # avoid EDecimal error
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = df[df[numeric_cols].notna().any(axis=1)]

    # Replace remaining usable NaNs with 0s
    df.fillna(0, inplace=True)

    # Drop string-based ips
    df = drop_columns(df, ['src', 'dst'])

    return df


def preprocess_file(pcap_path, batch_size,  label):
    try:
        logger.info("Starting batch preprocessing for: %s", pcap_path)

        file_obj = open(pcap_path, "rb")      # open file
        packet_reader = PcapReader(file_obj)        # open in reader

        df_list = []
        batch_rows = []
        last_pkt_time = None

        for pkt in tqdm_bar(packet_reader, desc="Preprocessing packets", unit="pkt"):
            row = extract_packet_fields(pkt, last_pkt_time)
            last_pkt_time = pkt.time
            batch_rows.append(row)

            if len(batch_rows) >= batch_size:
                df_batch = pd.DataFrame(batch_rows)
                if label is not None:
                    df_batch["anomaly"] = label
                df_clean = clean_dataframe(df_batch)
                df_list.append(df_clean)
                batch_rows = []
            
        # Handle remaining packets
        if batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            if label is not None:
                df_batch["anomaly"] = label
            df_clean = clean_dataframe(df_batch)
            df_list.append(df_clean)
            
        packet_reader.close()
        file_obj.close()        # close file

        df_final = pd.concat(df_list, ignore_index=True)  
        logger.info("Finished preprocessing. Total valid rows: %s", len(df_final))

        if df_final.empty:
            logger.error("Final preprocessed DataFrame is empty — no usable packets extracted.")

        return df_final

    except Exception as e:
        logger.exception("Failed to preprocess PCAP in batch mode: %s", e)
        return pd.DataFrame()


def run_preprocessor(args):
    batch_size = config['preprocessing']['batch_size']

    label = args.label or config['preprocessing']['label']
    pcap_input_path = args.input or config['preprocessing']['pcap_input']
    csv_output_path = args.output or config['preprocessing']['csv_output']

    df = preprocess_file(pcap_input_path, batch_size, label)

    # Save CSV
    output_csv_path = safe_save_path(csv_output_path)
    save_dataframe(df, output_csv_path)
