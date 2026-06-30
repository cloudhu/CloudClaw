'''Sector data multi-tier fallback chain.
Fallback: industry API -> concept API -> eastmoney HTTP -> pickle cache -> snapshot'''
import os, json, time, random, logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)
_http_session = None
_unavailable = set()
_last_req = 0.0
SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cache', 'sector_snapshot.json')

def _session():
    global _http_session
    if _http_session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        _http_session = requests.Session()
        r = Retry(total=2, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        _http_session.mount('http://', HTTPAdapter(max_retries=r))
        _http_session.mount('https://', HTTPAdapter(max_retries=r))
        _http_session.headers.update({'User-Agent':'Mozilla/5.0 Chrome/120.0.0.0'})
        _http_session.timeout = 15
    return _http_session

def _limit():
    global _last_req
    e = time.time() - _last_req
    if e < 0.8: time.sleep(0.8 - e + random.uniform(0,0.3))
    _last_req = time.time()

def _ak():
    import akshare as ak; return ak

def _save_snap(df):
    try:
        os.makedirs(os.path.dirname(SNAP), exist_ok=True)
        records = []
        for _, r in df.iterrows():
            d = {}
            for k, v in r.items():
                if hasattr(v, 'item'): d[k] = v.item()
                elif isinstance(v, float) and (pd.isna(v) or v != v): d[k] = None
                else: d[k] = v
            records.append(d)
        with open(SNAP, 'w', encoding='utf-8') as f:
            json.dump({'saved_at': datetime.now().isoformat(), 'data': records}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug('snapshot save fail: %s', e)

def _load_snap():
    try:
        if not os.path.exists(SNAP): return None
        with open(SNAP, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if d.get('data'): return pd.DataFrame(d['data'])
    except: pass
    return None

def reset():
    global _unavailable
    _unavailable.clear()

def _norm_cc(df):
    if df is not None and not df.empty:
        r = {}
        if '\u6982\u5ff5\u540d\u79f0' in df.columns: r['\u6982\u5ff5\u540d\u79f0'] = '\u677f\u5757\u540d\u79f0'
        if '\u6982\u5ff5\u4ee3\u7801' in df.columns: r['\u6982\u5ff5\u4ee3\u7801'] = '\u677f\u5757\u4ee3\u7801'
        if r: df = df.rename(columns=r)
    return df

def _http_list(stype):
    _limit()
    fs = 'm:90+t3' if stype == 'industry' else 'm:90+t2'
    url = 'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs={}&fields=f2,f3,f4,f12,f14,f20'.format(fs)
    r = _session().get(url, timeout=15)
    items = r.json().get('data', {}).get('diff', [])
    if not items: return pd.DataFrame()
    rows = [{'\u677f\u5757\u540d\u79f0': str(i.get('f14','')),
             '\u677f\u5757\u4ee3\u7801': str(i.get('f12','')),
             '\u6da8\u8dcc\u5e45': float(i.get('f3',0) or 0),
             '\u6700\u65b0\u4ef7': float(i.get('f2',0) or 0),
             '\u6da8\u8dcc\u989d': float(i.get('f4',0) or 0),
             '\u603b\u5e02\u503c': float(i.get('f20',0) or 0)} for i in items]
    return pd.DataFrame(rows)

def _http_stocks(bcode):
    _limit()
    url = 'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=b:{}%2Bf:!50&fields=f2,f3,f4,f5,f6,f8,f9,f12,f14,f20'.format(bcode)
    r = _session().get(url, timeout=15)
    items = r.json().get('data', {}).get('diff', [])
    if not items: return pd.DataFrame()
    rows = [{'\u4ee3\u7801': str(i.get('f12','')),
             '\u540d\u79f0': str(i.get('f14','')),
             '\u6700\u65b0\u4ef7': float(i.get('f2',0) or 0),
             '\u6da8\u8dcc\u5e45': float(i.get('f3',0) or 0),
             '\u6da8\u8dcc\u989d': float(i.get('f4',0) or 0),
             '\u6210\u4ea4\u91cf': float(i.get('f5',0) or 0),
             '\u6210\u4ea4\u989d': float(i.get('f6',0) or 0),
             '\u6362\u624b\u7387': float(i.get('f8',0) or 0),
             '\u5e02\u76c8\u7387-\u52a8\u6001': float(i.get('f9',0) or 0),
             '\u603b\u5e02\u503c': float(i.get('f20',0) or 0)} for i in items]
    return pd.DataFrame(rows)

def fetch_list(cache_mgr, force=False):
    if not force:
        c = cache_mgr.get('industry_list', max_age_hours=24)
        if c is not None: return c
    ak = _ak()
    sources = [
        ('ind_ak', lambda: (_limit(), ak.stock_board_industry_name_em())[1]),
        ('ind_cc', lambda: (_limit(), _norm_cc(ak.stock_board_concept_name_em()))[1]),
        ('ind_http_h', lambda: _http_list('industry')),
        ('ind_http_c', lambda: _http_list('concept')),
    ]
    for label, fetcher in sources:
        if label in _unavailable: continue
        try:
            df = fetcher()
            if df is not None and not df.empty:
                if label != sources[0][0]: logger.info('  -> sector list via %s', label)
                cache_mgr.set('industry_list', df)
                _save_snap(df)
                _unavailable.discard(label)
                return df
        except Exception as e:
            logger.warning('  sector list [%s] fail: %s', label, type(e).__name__)
            _unavailable.add(label)
    df = _load_snap()
    if df is not None and not df.empty:
        logger.info('  -> sector list from snapshot (all remote sources unavailable)')
        cache_mgr.set('industry_list', df)
        return df
    return pd.DataFrame()

def fetch_stocks(cache_mgr, industry, board_code=''):
    ckey = 'ind_stk_{}'.format(industry)
    c = cache_mgr.get(ckey, max_age_hours=6)
    if c is not None: return c
    ak = _ak()
    sources = [
        ('stk_ak', lambda: (_limit(), ak.stock_board_industry_cons_em(symbol=industry))[1]),
        ('stk_cc', lambda: (_limit(), ak.stock_board_concept_cons_em(symbol=industry))[1]),
    ]
    if board_code:
        sources.append(('stk_http', lambda: _http_stocks(board_code)))
    for label, fetcher in sources:
        if label in _unavailable: continue
        try:
            df = fetcher()
            if df is not None and not df.empty:
                if label != sources[0][0]: logger.info('  -> %s stocks via %s', industry, label)
                cache_mgr.set(ckey, df)
                _unavailable.discard(label)
                return df
        except Exception as e:
            logger.warning('  %s stocks [%s] fail: %s', industry, label, type(e).__name__)
    return pd.DataFrame()
