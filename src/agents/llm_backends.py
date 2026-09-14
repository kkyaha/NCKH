# -*- coding: utf-8 -*-
"""Factory backend LLM -- cho phep chay CUNG mot thi nghiem tren NHIEU model.

Ly do ton tai
-------------
RQ3/RQ5 truoc day chi chay tren MOT backend (`gemini-flash-lite-latest`).
Dieu do tao ra hai diem yeu tach biet:

1. *Tai lap*: `...-latest` la ALIAS TROI. Khi kiem tra ngay 2026-09-14, toan
   bo ho Gemini 2.x da bi go khoi API (`gemini-2.5-flash`, `gemini-2.0-flash`
   deu tra 404 "no longer available"). Nghia la so lieu do bang mot alias o
   thoi diem T KHONG chac tai lap duoc o thoi diem T+n, va nguoi doc khong co
   cach nao biet. Moi cau hinh duoi day vi vay deu ghi kem `pinned` (co phai
   snapshot co dinh khong) va `probed_on`.

2. *Hieu luc cua ket qua*: paper chung minh "LLM can guardrail" bang ti le vi
   pham cua baseline khong guard (PBVR 38%, SHR 91%, GMR 100%). Nhung do la
   ti le cua mot model LITE, re nhat thi truong. Mot reviewer se hoi lieu
   model frontier co hong nhu vay khong -- neu khong thi guardrail thanh
   thua. Bao dam cau truc (Proposition 1) DOC LAP backend, nhung ti le vi
   pham -- thu lam cho bao dam do co y nghia thay vi rong -- thi KHONG.
   Vi vay can chay it nhat mot model frontier va mot model thuoc nha cung
   cap khac.

Cach dung
---------
    from llm_backends import get_llm, available_backends, BACKENDS

    llm = get_llm('gpt-4o')          # hoac 'gemini-flash-lite', ...
    for name in available_backends(): ...   # chi nhung backend co API key
"""

import os
import warnings

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '.env'))
    load_dotenv(_ENV_PATH)
except Exception:  # dotenv khong bat buoc neu bien da nam trong shell env
    pass


# provider: 'google' | 'openai'
# tier:     'lite' | 'mid' | 'frontier'  -- dung de kiem tra gia thuyet
#           "model manh hon thi vi pham it hon"
# pinned:   True neu ten model la snapshot co dinh, False neu la alias troi
BACKENDS = {
    # --- Google ---
    'gemini-flash-lite': {
        'provider': 'google', 'model': 'gemini-flash-lite-latest',
        'tier': 'lite', 'pinned': False,
        'note': 'backend goc cua RQ3/RQ5 trong paper; alias troi',
    },
    'gemini-3.5-flash-lite': {
        'provider': 'google', 'model': 'gemini-3.5-flash-lite',
        'tier': 'lite', 'pinned': True,
    },
    'gemini-3.6-flash': {
        'provider': 'google', 'model': 'gemini-3.6-flash',
        'tier': 'mid', 'pinned': True,
    },
    'gemini-flash': {
        'provider': 'google', 'model': 'gemini-flash-latest',
        'tier': 'mid', 'pinned': False,
    },
    # Frontier DONG duy nhat co Free Tier (do 2026-09-14). Han muc la
    # GenerateContentInputTokensPerModelPerDay-FreeTier: mot tran token/NGAY,
    # KHONG phai tran phut -- da xac nhan bang cach thu lai sau 65s van 429.
    # => phai chia nho lan chay qua nhieu ngay, va --repeats=1.
    'gemini-3.1-pro': {
        'provider': 'google', 'model': 'gemini-3.1-pro-preview',
        'tier': 'frontier', 'pinned': True,
        'daily_token_cap': True,
        'note': 'free tier co tran token/ngay; can chay nhieu ngay',
    },
    # --- OpenAI (tra phi) ---
    'gpt-4o-mini': {
        'provider': 'openai', 'model': 'gpt-4o-mini-2024-07-18',
        'tier': 'lite', 'pinned': True,
    },
    'gpt-4o': {
        'provider': 'openai', 'model': 'gpt-4o-2024-11-20',
        'tier': 'frontier', 'pinned': True,
    },
    # --- Open-weight, phuc vu qua Groq ---
    # Uu diem quyet dinh cho tai lap: trong so CONG KHAI. Snapshot dong (Gemini
    # 2.x, cac ban gpt-4o) roi se bi go khoi API va khong ai chay lai duoc;
    # trong so mo thi khong bao gio bien mat. Ghi kem nha cung cap suy luan vi
    # ho co the luong tu hoa -- hanh vi co the lech nhe so voi trong so goc.
    'gpt-oss-20b': {
        'provider': 'groq', 'model': 'openai/gpt-oss-20b',
        'tier': 'small', 'pinned': True, 'open_weights': True,
        'served_by': 'Groq (quantization as applied by provider)',
    },
    'gpt-oss-120b': {
        'provider': 'groq', 'model': 'openai/gpt-oss-120b',
        'tier': 'large', 'pinned': True, 'open_weights': True,
        'served_by': 'Groq (quantization as applied by provider)',
    },
    # KHONG dua 'groq/compound' vao: do la mot HE THONG AGENTIC co san tool-use,
    # khong phai mot LLM tran. Dung no lam baseline "LLM khong guard" se lam
    # nhiem ket qua, vi no da co scaffolding rieng -- dung thu ma bai bao dang
    # do luong tac dung.
}

_ENV_VAR = {'google': 'GOOGLE_API_KEY', 'openai': 'OPENAI_API_KEY',
            'groq': 'GROQ_API_KEY'}


def has_credentials(provider: str) -> bool:
    return bool(os.getenv(_ENV_VAR.get(provider, '')))


def available_backends(tier: str = None, provider: str = None) -> list:
    """Ten cac backend co the goi duoc ngay (da co API key)."""
    out = []
    for name, cfg in BACKENDS.items():
        if not has_credentials(cfg['provider']):
            continue
        if tier and cfg['tier'] != tier:
            continue
        if provider and cfg['provider'] != provider:
            continue
        out.append(name)
    return out


def get_llm(backend: str, temperature: float = 0.2, max_retries: int = 0):
    """Tra ve mot LangChain chat model theo ten backend trong BACKENDS."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Backend '{backend}' khong co. Chon trong: {sorted(BACKENDS)}")
    cfg = BACKENDS[backend]
    provider, model = cfg['provider'], cfg['model']

    if not has_credentials(provider):
        raise RuntimeError(
            f"Thieu {_ENV_VAR[provider]} cho backend '{backend}'. "
            f"Them vao .env hoac export trong shell.")

    if not cfg['pinned']:
        warnings.warn(
            f"Backend '{backend}' dung alias troi ('{model}'): ket qua co the "
            f"khong tai lap duoc ve sau. Uu tien snapshot co ngay khi bao cao.",
            stacklevel=2)

    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      max_retries=max_retries)
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature,
                          max_retries=max_retries)
    if provider == 'groq':
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature,
                        max_retries=max_retries)
    raise ValueError(f"Provider khong ho tro: {provider}")


def backend_metadata(backend: str) -> dict:
    """Metadata de ghi kem MOI dong ket qua -- phuc vu tai lap."""
    cfg = dict(BACKENDS[backend])
    cfg['backend'] = backend
    return cfg


if __name__ == '__main__':
    print(f"{'backend':24s} {'provider':9s} {'tier':9s} {'pinned':7s} {'open-w':7s} available")
    for name, cfg in BACKENDS.items():
        print(f"{name:24s} {cfg['provider']:9s} {cfg['tier']:9s} "
              f"{str(cfg['pinned']):7s} {str(cfg.get('open_weights', False)):7s} "
              f"{has_credentials(cfg['provider'])}")
    print(f"\nSan sang chay: {available_backends()}")
