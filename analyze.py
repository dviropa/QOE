# python analyze.py --window-size 10 --sample-duration 1 --pcap-path /Users/eyal/PhD/github/qoe-research/exp/000/pcaps --short-window-path /Users/eyal/PhD/github/qoe-research/band --exp-csv-path /Users/eyal/PhD/github/qoe-research/exp/17-May-2024/summary.csv
import sys
import tempfile
from subprocess import run
import argparse
from nfstream import NFStreamer, NFPlugin
from time import time
import os
from PingLatency import PingLatency
import pandas as pd

def extract_last_int(string):
    if '_' in string:
        return int(string[string.rindex('_') + 1:])
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('--window-size', action='store', help='clone dir path', type=int, default=10)
    parser.add_argument('--sample-duration', action='store', help='sample duration', type=int, default=1)
    parser.add_argument('--pcap-dir', action='store', help='pcap dir path',default='pcap/internal_session_20250925_1728/20250925_1728(2)')
    parser.add_argument('--short-window-path', action='store', help='pcap file path')
    parser.add_argument('--exp-csv-path', action='store', help='experiments csv path')

    args = parser.parse_args()
    window_size = args.window_size
    sample_duration = args.sample_duration
    pcap_dir = args.pcap_dir
    short_window_path = args.short_window_path
    exp_csv_path = args.exp_csv_path
    
    global_pcap_data = []
    long_window_df_list = []
    if not os.path.isdir(pcap_dir):
        print(f'pcap_dir {pcap_dir} not found')
        sys.exit(1)
    df = pd.read_csv(exp_csv_path)
    for exp_id in os.listdir(pcap_dir):
        pcap_path = os.path.join(pcap_dir, exp_id)
        print(f'pcap_filename: {pcap_path}')
        if pcap_path.endswith('.pcapng'):
            exp_id = os.path.basename(pcap_path).split('.')[0]
            exp_row = df.loc[df['exp_id'] == int(exp_id)]
            if len(exp_row) == 1:
                ping = int(exp_row.iloc[0].avg_ping_rate)
                impact = int(exp_row.iloc[0].impact)
                self_rate_qoe_score = int(exp_row.iloc[0].self_rate_qoe_score)
                self_rate_player_performance = int(exp_row.iloc[0].self_rate_player_performance)
                rolling_short_df = get_pcap_features(pcap_path, window_size, sample_duration, impact=impact, self_rate_player_performance=self_rate_player_performance, self_rate_qoe_score=self_rate_qoe_score, ping=ping, day=0, idx=0, short_window_path=short_window_path, exp_id=exp_id)
                long_window_df_list.append(rolling_short_df)

    window_df = pd.concat(long_window_df_list)
    print(f'pre filters {len(window_df)}')
    window_df = window_df[(window_df['dst2src_pps'] < 1000) & (window_df['src2dst_pps'] < 1000)]    
    print(f'post pps filter {len(window_df)}')
    os.makedirs(short_window_path, exist_ok=True)
    output_path = os.path.join(short_window_path, f'window_{window_size}_{int(time())}.csv')
    print(f'output_path: {output_path}')
    window_df.to_csv(output_path)

def get_pcap_features(filepath, window_size, sample_duration, ping=None, day=None, idx=None, impact=None, self_rate_player_performance=None, self_rate_qoe_score=None, short_window_path='/Users/eyal/PhD/qoe/short_10sec_window', exp_id=0):
    short_window_pcap_data = []
    print(filepath)

    temp_dir = tempfile.mkdtemp()
    file_prefix = os.path.join(temp_dir, f'{ping}_{ping}_{day}_{idx}')
    run_output = run(['editcap', '-i', f'{sample_duration}', '-F', 'pcapng', filepath, file_prefix]) # 1 second window
    print('editcap run_output:', run_output)
    sorted_list = sorted(os.listdir(temp_dir), key=extract_last_int)
    for pktIdx, filename in enumerate(sorted_list):
        short_pcap_path = os.path.join(temp_dir, filename)
        window_features = extract_raw_features(short_pcap_path, ping=ping, day=day, idx=idx, pktIdx=pktIdx, impact=impact, self_rate_player_performance=self_rate_player_performance, self_rate_qoe_score=self_rate_qoe_score, to_dict=True)
        if window_features:
            short_window_pcap_data.append(window_features)
    short_df = pd.DataFrame.from_dict(short_window_pcap_data)
    # path = os.path.join(short_window_path, os.path.basename(filepath).replace('.pcapng', '.csv'))
    # short_df.to_csv(path, index=False)
    rolling_short_df = pd.DataFrame()
    rolling_short_df['impact'] = short_df['impact']
    rolling_short_df['self_rate_qoe_score'] = short_df['self_rate_qoe_score']
    rolling_short_df['self_rate_player_performance'] = short_df['self_rate_player_performance']
    rolling_short_df['src2dst_bytes'] = short_df['src2dst_bytes']
    rolling_short_df['dst2src_bytes'] = short_df['dst2src_bytes']
    rolling_short_df['ping'] = short_df['ping']
    rolling_short_df['day'] = short_df['day']
    rolling_short_df['dst_port'] = short_df['dst_port']
    rolling_short_df['pktIdx'] = short_df['pktIdx']
    rolling_short_df['exp_id'] = exp_id
    for col in ('src2dst_avg_pkt_size', 'dst2src_avg_pkt_size', 'dst2src_pps', 'src2dst_pps', 'pkt_dir_ratio'): # removed 'i'
        rolling_short_df[col] = short_df[col]
        rolling_short_df[f"{col}_mean"] = short_df[col].rolling(window=window_size).mean()
        rolling_short_df[f"{col}_median"] = short_df[col].rolling(window=window_size).median()
        rolling_short_df[f"{col}_std"] = short_df[col].rolling(window=window_size).std()
        rolling_short_df[f"{col}_min"] = short_df[col].rolling(window=window_size).min()
        rolling_short_df[f"{col}_max"] = short_df[col].rolling(window=window_size).max()
    # rolling_short_df.dropna(inplace=True)
    return rolling_short_df

def extract_raw_features(filepath, ping=None, day=None, idx=None, pktIdx=None, impact=None, self_rate_player_performance=None, self_rate_qoe_score=None, to_dict=True):
    d = None
    if os.path.isfile(filepath):
        df = NFStreamer(source=filepath, udps=[PingLatency()], bpf_filter='udp and (portrange 7000-10000)').to_pandas()
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            # print('no data to process')
            return None
        df.sort_values(by='src2dst_bytes', ascending=False, inplace=True)
        df['impact'] = impact
        df['self_rate_qoe_score'] = self_rate_qoe_score
        df['self_rate_player_performance'] = self_rate_player_performance
        df['ping'] = ping
        df['day'] = day
        df['idx'] = idx
        df['pktIdx'] = pktIdx
        if df.iloc[0]['dst_port'] > 10000:
            df.rename(columns={
                'src2dst_bytes': 'dst2src_bytes',
                'dst2src_bytes': 'src2dst_bytes',
                'src2dst_packets': 'dst2src_packets',
                'dst2src_packets': 'src2dst_packets',
                'dst_port': 'src_port',
                'src_port': 'dst_port',
            }, inplace=True)
        df['src2dst_avg_pkt_size'] = df['src2dst_bytes'] / df['src2dst_packets']
        df['dst2src_avg_pkt_size'] = df['dst2src_bytes'] / df['dst2src_packets']
        df['pkt_dir_ratio'] = df['src2dst_packets'] / df['dst2src_packets']
        df['src2dst_duration'] = df.apply(lambda x: max(x['src2dst_last_seen_ms'] - x['src2dst_first_seen_ms'], 1), axis=1)
        df['dst2src_duration'] = df.apply(lambda x: max(x['dst2src_last_seen_ms'] - x['dst2src_first_seen_ms'], 1), axis=1)
        df['src2dst_pps'] = df.apply(lambda x: x['src2dst_packets'] / x['src2dst_duration'] * 1000, axis=1)
        df['dst2src_pps'] = df.apply(lambda x: x['dst2src_packets'] / x['dst2src_duration'] * 1000, axis=1)
        
        if to_dict:
            d = df.iloc[0] \
                    [['src2dst_bytes', 'dst2src_bytes', 'impact', 'self_rate_player_performance', 'self_rate_qoe_score', 'ping', 'day', 'idx', 'dst_port', 'pktIdx', 'src2dst_avg_pkt_size', 'src2dst_pps', 'dst2src_avg_pkt_size', 'dst2src_pps', 'pkt_dir_ratio']].to_dict()
        else:
            d = df[['src2dst_bytes', 'dst2src_bytes', 'src2dst_avg_pkt_size', 'src2dst_pps', 'dst2src_avg_pkt_size', 'dst2src_pps']].head(1) 
    return d

if __name__ == '__main__':
    main()

