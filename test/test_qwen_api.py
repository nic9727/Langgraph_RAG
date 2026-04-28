"""
Qwen 大模型 API 接口测试脚本 (test/test_qwen_api.py)

本脚本用于测试阿里云百炼平台 qwen3.5-plus 大模型的 API 连通性和各项功能。
包含以下测试项：
  0. 网络诊断     — SSL 版本检查、DNS、TCP、HTTPS、HTTP 逐层排查
  1. 基础连通性测试 — 验证 API Key 和 OpenAI SDK 调用是否正常
  2. 普通对话测试   — 测试 chat() 函数的基本问答能力
  3. JSON 格式输出测试 — 测试 chat_with_json() 的 JSON 解析功能
  4. 流式输出测试   — 测试 chat_stream() 的逐块输出功能
  5. 多轮对话测试   — 测试带 history 参数的多轮对话上下文保持
  6. 自定义参数测试 — 测试 temperature、max_tokens 等参数的效果

使用方式：
  cd Langragh_RAG
  python test/test_qwen_api.py              # 运行全部测试
  python test/test_qwen_api.py --diag       # 只运行网络诊断
  python test/test_qwen_api.py --no-verify  # 跳过 SSL 验证（解决 SSL 握手问题）
"""

import sys
import os
import time
import json
import socket
import ssl
import traceback
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# 将项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# ===================== API 配置（与项目一致） =====================
API_KEY = "sk-6eab71b446594503ab07642a4d2f9cce"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

# 完整的 chat completions 端点
FULL_ENDPOINT = f"{BASE_URL}/chat/completions"
HOST = "dashscope.aliyuncs.com"
PORT = 443

# 是否跳过 SSL 验证（通过 --no-verify 参数控制）
SKIP_SSL_VERIFY = "--no-verify" in sys.argv


def print_separator(title: str):
    """打印分隔线和测试标题。"""
    print("\n" + "=" * 60)
    print(f"  测试 {title}")
    print("=" * 60)


def _get_unverified_ssl_context():
    """创建一个不验证证书的 SSL 上下文（仅用于调试 SSL 问题）。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def test_0_network_diagnosis():
    """
    测试 0：网络诊断 — 逐层排查连接问题。
    依次测试 Python/OpenSSL 版本、DNS、TCP、HTTPS、HTTP、代理。
    """
    print_separator("0: 网络诊断（逐层排查）")
    all_ok = True

    # ----- 第 0 步：Python 和 OpenSSL 版本 -----
    print(f"\n  [0/6] Python 和 OpenSSL 版本信息")
    print(f"        Python: {sys.version}")
    print(f"        OpenSSL: {ssl.OPENSSL_VERSION}")
    openssl_ver = ssl.OPENSSL_VERSION
    if "1.0" in openssl_ver or "1.1.0" in openssl_ver:
        print(f"        ⚠️  OpenSSL 版本较旧，可能导致 TLS 握手失败！")
        print(f"        → 建议运行: conda update openssl")
    else:
        print(f"        ✅ OpenSSL 版本正常")

    # ----- 第 1 步：DNS 解析 -----
    print(f"\n  [1/6] DNS 解析 {HOST} ...")
    try:
        ip_list = socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM)
        ips = list(set(addr[4][0] for addr in ip_list))
        print(f"        ✅ 解析成功，IP: {', '.join(ips)}")
    except socket.gaierror as e:
        print(f"        ❌ DNS 解析失败: {e}")
        print(f"        → 请检查网络连接或 DNS 设置")
        return False

    # ----- 第 2 步：TCP 连接 -----
    print(f"\n  [2/6] TCP 连接 {HOST}:{PORT} ...")
    try:
        sock = socket.create_connection((HOST, PORT), timeout=10)
        print(f"        ✅ TCP 连接成功")
        sock.close()
    except (socket.timeout, socket.error) as e:
        print(f"        ❌ TCP 连接失败: {e}")
        print(f"        → 可能是防火墙/代理阻止了 443 端口的出站连接")
        return False

    # ----- 第 3 步：SSL/TLS 握手（标准验证） -----
    print(f"\n  [3/6] SSL/TLS 握手（标准模式）...")
    try:
        context = ssl.create_default_context()
        sock = socket.create_connection((HOST, PORT), timeout=10)
        ssock = context.wrap_socket(sock, server_hostname=HOST)
        print(f"        ✅ TLS 握手成功，协议: {ssock.version()}")
        ssock.close()
    except ssl.SSLEOFError as e:
        print(f"        ❌ SSL 握手失败 (SSLEOFError): {e}")
        print(f"        → 这通常是以下原因之一：")
        print(f"          1. 代理/VPN 正在拦截 HTTPS 流量（最常见）")
        print(f"          2. Anaconda 的 OpenSSL 版本过旧")
        print(f"          3. 企业防火墙进行了 SSL 检查")
        print(f"        → 修复建议：")
        print(f"          a. 关闭代理/VPN 后重试")
        print(f"          b. conda update openssl ca-certificates certifi")
        print(f"          c. 使用 --no-verify 参数跳过 SSL 验证测试")
        all_ok = False

        # 尝试不验证证书的握手
        print(f"\n  [3b/6] SSL/TLS 握手（跳过证书验证）...")
        try:
            ctx_noverify = _get_unverified_ssl_context()
            sock2 = socket.create_connection((HOST, PORT), timeout=10)
            ssock2 = ctx_noverify.wrap_socket(sock2, server_hostname=HOST)
            print(f"         ✅ 不验证证书时握手成功！协议: {ssock2.version()}")
            print(f"         → 说明问题出在证书验证上，可能是代理替换了证书")
            ssock2.close()
        except Exception as e2:
            print(f"         ❌ 即使跳过验证也失败: {e2}")
            print(f"         → 说明问题不是证书，而是网络层面的阻断")

    except ssl.SSLError as e:
        print(f"        ❌ SSL 错误: {e}")
        all_ok = False
    except Exception as e:
        print(f"        ❌ 握手异常: {e}")
        all_ok = False

    # ----- 第 4 步：HTTP POST 请求（urllib） -----
    print(f"\n  [4/6] HTTP POST 测试（urllib 直接请求）...")
    try:
        payload = json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }).encode("utf-8")

        req = Request(
            FULL_ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )

        # 根据参数决定是否跳过 SSL 验证
        ssl_ctx = _get_unverified_ssl_context() if SKIP_SSL_VERIFY else None

        start = time.time()
        resp = urlopen(req, timeout=30, context=ssl_ctx)
        elapsed = time.time() - start
        body = resp.read().decode("utf-8")
        data = json.loads(body)

        print(f"        ✅ HTTP 请求成功！状态码: {resp.status}")
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "(空)")
        print(f"        模型返回: {content[:50]}")
        print(f"        耗时: {elapsed:.2f} 秒")

    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"        ⚠️  HTTP 错误 {e.code}: {e.reason}")
        print(f"        响应体: {body[:300]}")
        if e.code == 401:
            print(f"        → API Key 无效或已过期")
        elif e.code == 403:
            print(f"        → 没有访问权限，请检查 API Key 的权限设置")
        elif e.code == 429:
            print(f"        → 请求频率超限，请稍后重试")
        all_ok = False
    except URLError as e:
        print(f"        ❌ 请求失败: {e.reason}")
        if "SSL" in str(e.reason) or "EOF" in str(e.reason):
            print(f"        → SSL 相关错误，尝试加 --no-verify 参数重跑：")
            print(f"          python test/test_qwen_api.py --no-verify")
        all_ok = False
    except Exception as e:
        print(f"        ❌ 请求异常: {e}")
        traceback.print_exc()
        all_ok = False

    # ----- 第 5 步：检查代理设置 -----
    print(f"\n  [5/6] 代理环境变量检查 ...")
    proxy_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "no_proxy", "NO_PROXY", "ALL_PROXY", "all_proxy"]
    found_proxy = False
    for var in proxy_vars:
        val = os.environ.get(var)
        if val:
            print(f"        {var} = {val}")
            found_proxy = True
    if not found_proxy:
        print(f"        未检测到代理环境变量")
    else:
        print(f"        ⚠️  检测到代理设置，如果上面连接失败，可能是代理导致的")
        print(f"        → 尝试临时取消代理后重试:")
        print(f"          Windows CMD:  set HTTPS_PROXY=")
        print(f"          PowerShell:   $env:HTTPS_PROXY=''")

    # ----- 第 6 步：openai 库版本 -----
    print(f"\n  [6/6] openai 库版本检查 ...")
    try:
        import openai
        print(f"        openai: {openai.__version__}")
        import httpx
        print(f"        httpx: {httpx.__version__}")
    except ImportError as e:
        print(f"        ⚠️  {e}")

    if all_ok:
        print(f"\n  ✅ 网络诊断全部通过！API 可以正常连接。")
    else:
        print(f"\n  ⚠️  部分检查未通过，请根据上面的提示排查。")
    return all_ok


def _get_openai_client():
    """获取 OpenAI 客户端，根据 --no-verify 参数决定是否跳过 SSL 验证。"""
    from openai import OpenAI
    import httpx

    if SKIP_SSL_VERIFY:
        http_client = httpx.Client(verify=False)
        return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)
    else:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def test_1_basic_connection():
    """测试 1：基础连通性 — 使用 OpenAI SDK 验证调用。"""
    print_separator("1: 基础连通性（OpenAI SDK）")

    if SKIP_SSL_VERIFY:
        print("  (已启用 --no-verify 模式，跳过 SSL 证书验证)")

    try:
        client = _get_openai_client()
        start = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": '你好，请回复"测试成功"两个字'}],
            max_tokens=50,
            temperature=0.1,
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content
        print(f"  ✅ 连接成功！")
        print(f"  模型: {response.model}")
        print(f"  回复: {content}")
        print(f"  耗时: {elapsed:.2f} 秒")
        print(f"  Token 用量: prompt={response.usage.prompt_tokens}, "
              f"completion={response.usage.completion_tokens}, "
              f"total={response.usage.total_tokens}")
        return True
    except Exception as e:
        print(f"  ❌ 连接失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_2_chat():
    """测试 2：普通对话 — 测试 chat() 函数。"""
    print_separator("2: 普通对话（chat）")

    from llm.qwen_client import chat

    try:
        start = time.time()
        answer = chat(
            user_message="请用一句话解释什么是自然语言处理（NLP）？",
            system_prompt="你是一位计算机科学教授，请简洁地回答问题。",
            temperature=0.3,
            max_tokens=200,
        )
        elapsed = time.time() - start

        print(f"  ✅ 调用成功！")
        print(f"  问题: 请用一句话解释什么是自然语言处理（NLP）？")
        print(f"  回答: {answer}")
        print(f"  耗时: {elapsed:.2f} 秒")
        return True
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_3_chat_with_json():
    """测试 3：JSON 格式输出 — 测试 chat_with_json() 函数。"""
    print_separator("3: JSON 格式输出（chat_with_json）")

    from llm.qwen_client import chat_with_json

    try:
        start = time.time()
        result = chat_with_json(
            user_message='请将以下句子分类为"问候"、"提问"或"陈述"，并给出置信度。句子："今天天气怎么样？"',
            system_prompt="你是一个文本分类器。请以JSON格式回答，包含 category 和 confidence 两个字段。",
            temperature=0.1,
        )
        elapsed = time.time() - start

        print(f"  ✅ 调用成功！")
        print(f"  返回类型: {type(result).__name__}")
        print(f"  返回内容: {result}")
        if "raw" in result:
            print(f"  ⚠️  注意: JSON 解析失败，返回了原始文本")
        else:
            print(f"  ✅ JSON 解析成功")
        print(f"  耗时: {elapsed:.2f} 秒")
        return True
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_4_chat_stream():
    """测试 4：流式输出 — 测试 chat_stream() 函数。"""
    print_separator("4: 流式输出（chat_stream）")

    from llm.qwen_client import chat_stream

    try:
        start = time.time()
        print(f"  问题: 请用3个要点介绍Python语言的特点")
        print(f"  流式回答: ", end="", flush=True)

        chunk_count = 0
        full_text = ""
        for chunk in chat_stream(
            user_message="请用3个要点介绍Python语言的特点，每个要点一句话。",
            system_prompt="你是一位编程老师，请简洁回答。",
            temperature=0.5,
            max_tokens=300,
        ):
            print(chunk, end="", flush=True)
            full_text += chunk
            chunk_count += 1
        elapsed = time.time() - start
        print()
        print(f"\n  ✅ 流式输出成功！")
        print(f"  共收到 {chunk_count} 个 chunk")
        print(f"  总字符数: {len(full_text)}")
        print(f"  耗时: {elapsed:.2f} 秒")
        return True
    except Exception as e:
        print(f"\n  ❌ 调用失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_5_multi_turn():
    """测试 5：多轮对话 — 测试 history 参数的上下文保持能力。"""
    print_separator("5: 多轮对话（history）")

    from llm.qwen_client import chat

    try:
        print("  --- 第 1 轮 ---")
        q1 = "我叫小明，我是计算机科学专业的学生。"
        a1 = chat(user_message=q1, temperature=0.3, max_tokens=200)
        print(f"  用户: {q1}")
        print(f"  助手: {a1}")

        print("\n  --- 第 2 轮 ---")
        q2 = "请问我叫什么名字？我是什么专业的？"
        history = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
        ]
        a2 = chat(user_message=q2, history=history, temperature=0.1, max_tokens=200)
        print(f"  用户: {q2}")
        print(f"  助手: {a2}")

        if "小明" in a2 and "计算机" in a2:
            print(f"\n  ✅ 多轮对话上下文保持正常！模型记住了用户信息。")
        else:
            print(f"\n  ⚠️  模型可能未完全记住上下文，请查看回答内容。")
        return True
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_6_custom_params():
    """测试 6：自定义参数 — 测试不同 temperature 的效果。"""
    print_separator("6: 自定义参数（temperature 对比）")

    from llm.qwen_client import chat

    try:
        question = "用一个词形容人工智能的未来。"

        print(f"  问题: {question}")
        print(f"\n  temperature=0.1（低随机性）:")
        for i in range(3):
            answer = chat(user_message=question, temperature=0.1, max_tokens=20)
            print(f"    第{i+1}次: {answer.strip()}")

        print(f"\n  temperature=1.5（高随机性）:")
        for i in range(3):
            answer = chat(user_message=question, temperature=1.5, max_tokens=20)
            print(f"    第{i+1}次: {answer.strip()}")

        print(f"\n  ✅ 参数测试完成！低温度回答应更一致，高温度回答应更多样。")
        return True
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试。"""
    print("+" + "-" * 58 + "+")
    print("|  Qwen-Plus 大模型 API 接口测试" + " " * 27 + "|")
    print(f"|  模型: {MODEL_NAME}" + " " * (50 - len(MODEL_NAME)) + "|")
    print(f"|  Base URL: {BASE_URL}" + " " * 4 + "|")
    print(f"|  完整端点: {FULL_ENDPOINT}" + " " * (46 - len(FULL_ENDPOINT)) + "|")
    if SKIP_SSL_VERIFY:
        print("|  ⚠️  SSL 验证: 已禁用 (--no-verify)" + " " * 20 + "|")
    print("+" + "-" * 58 + "+")

    diag_only = "--diag" in sys.argv

    tests = [
        ("网络诊断", test_0_network_diagnosis),
    ]

    if not diag_only:
        tests.extend([
            ("基础连通性", test_1_basic_connection),
            ("普通对话", test_2_chat),
            ("JSON 格式输出", test_3_chat_with_json),
            ("流式输出", test_4_chat_stream),
            ("多轮对话", test_5_multi_turn),
            ("自定义参数", test_6_custom_params),
        ])

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print(f"\n  用户中断测试")
            results[name] = False
            break
        except Exception as e:
            print(f"\n  ❌ 未预期的错误: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = False

        # 如果网络诊断失败且不是仅诊断模式，提示用户
        if name == "网络诊断" and not results[name] and not diag_only:
            print("\n  ⚠️  网络诊断未通过，后续 API 测试可能也会失败。")
            print("  提示: 可以尝试 python test/test_qwen_api.py --no-verify")
            user_input = input("  是否继续运行后续测试？(y/n): ").strip().lower()
            if user_input != "y":
                break

    # 打印测试汇总
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    passed = 0
    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status}  {name}")
        if success:
            passed += 1
    total = len(results)
    print(f"\n  总计: {passed}/{total} 通过")

    if passed < total:
        print("\n  排查建议:")
        print("  1. 关闭代理/VPN 后重试（最常见原因）")
        print("  2. 更新 SSL: conda update openssl ca-certificates certifi")
        print("  3. 升级 HTTP 库: pip install --upgrade httpx httpcore openai")
        print("  4. 使用 --no-verify 跳过 SSL 验证测试连通性:")
        print("     python test/test_qwen_api.py --no-verify")
        print("  5. 检查 API Key: 登录 https://bailian.console.aliyun.com/ 确认")
        print("  6. 用 curl 手动测试:")
        print(f'     curl -X POST "{FULL_ENDPOINT}" ^')
        print(f'       -H "Authorization: Bearer {API_KEY[:10]}...{API_KEY[-4:]}" ^')
        print(f'       -H "Content-Type: application/json" ^')
        print(f'       -d "{{\\"model\\":\\"{MODEL_NAME}\\",\\"messages\\":[{{\\"role\\":\\"user\\",\\"content\\":\\"hi\\"}}]}}"')

    print("=" * 60)


if __name__ == "__main__":
    main()
