from pathlib import Path
import json
import textwrap
import pandas as pd
# from caas_jupyter_tools import display_dataframe_to_user

import tiktoken

# -----------------------------
# Sample corpus: 5 short English articles with parallel representations
# -----------------------------
samples = [
    {
        "id": "A1",
        "topic": "SSE middleware buffering",
        "english": (
            "A workflow system can fail when a middleware buffers streaming responses. "
            "In one case, ordinary polling worked, but server-sent events appeared to hang. "
            "The cause was not the workflow engine itself. The real issue was that a "
            "response-rewriting middleware held back the body until the whole response finished. "
            "The fix was simple: apply the rewriting logic only to MCP routes and let other routes "
            "pass through untouched. After that change, SSE streamed normally again."
        ),
        "wenyan": (
            "工作流系统，中间件缓冲流式响应，则可失效。尝有一例：轮询常通，而SSE若悬。"
            "其故不在引擎，实在改写响应之中间件，执其体而俟全响应毕。其解甚简：惟MCP诸路施改写，"
            "余路悉透传。既改，SSE复常流。"
        ),
        "graph": textwrap.dedent("""\
            N:
            a=workflow_system
            b=buffering_middleware
            c=streaming_response
            d=polling
            e=SSE
            f=workflow_engine
            g=response_rewrite_middleware
            h=body_held_until_response_end
            i=fix
            j=MCP_routes
            k=other_routes
            l=SSE_restored

            E:
            1:(b)-buffers->(c)
            2:(a)-fails_when->(1)
            3:(d)-status->worked
            4:(e)-status->hung
            5:(f)-not_cause_of->problem
            6:(g)-holds_back->body
            7:(g)-until->response_end
            8:(i)-apply_only_to->(j)
            9:(i)-pass_through->(k)
            10:(i)-restores->(l)
        """).strip(),
        "hypergraph": textwrap.dedent("""\
            N:
            a=workflow_system
            b=buffering_middleware
            c=streaming_response
            d=failure
            e=polling
            f=worked
            g=SSE
            h=hung
            i=workflow_engine
            j=problem
            k=response_rewrite_middleware
            l=body
            m=response_end
            n=fix
            o=MCP_routes
            p=other_routes
            q=SSE_streaming_restored

            H:
            H1:{a,b,c}-causes->{d}
            H2:{e}-status->{f}
            H3:{g}-status->{h}
            H4:{i}-not_cause_of->{j}
            H5:{k}-holds_back->{l}
            H6:{k}-until->{m}
            H7:{n}-apply_only_to->{o}
            H8:{n}-pass_through->{p}
            H9:{n}-restores->{q}
        """).strip(),
        "wenyan_hypergraph": textwrap.dedent("""\
            N:
            甲=工作流系统
            乙=缓冲中间件
            丙=流式响应
            丁=失效
            戊=轮询
            己=通
            庚=SSE
            辛=悬
            壬=引擎
            癸=问题
            子=改写中间件
            丑=体
            寅=响应毕
            卯=解
            辰=MCP诸路
            巳=余路
            午=SSE复常流

            H:
            一:{甲,乙,丙}-致->{丁}
            二:{戊}-状->{己}
            三:{庚}-状->{辛}
            四:{壬}-非因->{癸}
            五:{子}-执后->{丑}
            六:{子}-俟->{寅}
            七:{卯}-惟施->{辰}
            八:{卯}-透传->{巳}
            九:{卯}-复->{午}
        """).strip(),
    },
    {
        "id": "A2",
        "topic": "Deployment outage after secret rotation",
        "english": (
            "A deployment pipeline failed after a secret rotation. Builds still started, "
            "but image pushes to the registry were rejected. The application code was not the cause. "
            "The real problem was an expired service principal credential used by the publish step. "
            "The fix was to update the secret in the CI environment, re-authenticate, and rerun only "
            "the failed stage. After that, deployment completed successfully."
        ),
        "wenyan": (
            "部署流水线，密钥轮换后失。构建犹起，而镜像推仓为拒。其故不在应用代码，"
            "实在发布步骤所用服务主体凭证已逾期。其解：更新CI环境之秘钥，重验其身，"
            "独重行其败段。既而部署遂成。"
        ),
        "graph": textwrap.dedent("""\
            N:
            a=deployment_pipeline
            b=secret_rotation
            c=builds
            d=registry_push
            e=application_code
            f=expired_service_principal_credential
            g=publish_step
            h=fix
            i=CI_environment
            j=re_authenticate
            k=rerun_failed_stage
            l=deployment_success

            E:
            1:(a)-failed_after->(b)
            2:(c)-status->started
            3:(d)-status->rejected
            4:(e)-not_cause_of->problem
            5:(f)-used_by->(g)
            6:(f)-caused->problem
            7:(h)-update_secret_in->(i)
            8:(h)-includes->(j)
            9:(h)-includes->(k)
            10:(h)-restores->(l)
        """).strip(),
        "hypergraph": textwrap.dedent("""\
            N:
            a=deployment_pipeline
            b=secret_rotation
            c=failure
            d=builds
            e=started
            f=registry_push
            g=rejected
            h=application_code
            i=problem
            j=expired_service_principal_credential
            k=publish_step
            l=fix
            m=CI_environment
            n=re_authenticate
            o=rerun_failed_stage
            p=deployment_success

            H:
            H1:{a,b}-precedes->{c}
            H2:{d}-status->{e}
            H3:{f}-status->{g}
            H4:{h}-not_cause_of->{i}
            H5:{j,k}-causes->{i}
            H6:{l}-update_secret_in->{m}
            H7:{l}-includes->{n,o}
            H8:{l}-restores->{p}
        """).strip(),
        "wenyan_hypergraph": textwrap.dedent("""\
            N:
            甲=部署流水线
            乙=密钥轮换
            丙=失
            丁=构建
            戊=起
            己=推仓
            庚=拒
            辛=应用代码
            壬=问题
            癸=逾期凭证
            子=发布步骤
            丑=解
            寅=CI环境
            卯=重验
            辰=重行败段
            巳=部署成

            H:
            一:{甲,乙}-后见->{丙}
            二:{丁}-状->{戊}
            三:{己}-状->{庚}
            四:{辛}-非因->{壬}
            五:{癸,子}-致->{壬}
            六:{丑}-更新秘钥于->{寅}
            七:{丑}-兼->{卯,辰}
            八:{丑}-复->{巳}
        """).strip(),
    },
    {
        "id": "A3",
        "topic": "Feature launch and cache invalidation",
        "english": (
            "A new search feature launched with a stale cache problem. Users could submit queries, "
            "but fresh documents did not appear in results for several minutes. The ranking model was "
            "not the root cause. The real issue was that cache invalidation ran on a delayed schedule "
            "instead of on write. The fix was to invalidate affected keys during ingestion and keep the "
            "scheduled sweep as a backup. After that, new documents appeared almost immediately."
        ),
        "wenyan": (
            "新检索功能既发，而有陈缓存之患。用户虽可发问，新文数分钟不见于果。"
            "排序模型非其本因，实在缓存失效不随写入而行，乃迟时程而作。其解：摄入时即废受及诸键，"
            "而留定时清扫为备。既改，新文几即见。"
        ),
        "graph": textwrap.dedent("""\
            N:
            a=new_search_feature
            b=stale_cache_problem
            c=users
            d=queries
            e=fresh_documents
            f=results
            g=ranking_model
            h=cache_invalidation
            i=delayed_schedule
            j=on_write
            k=fix
            l=affected_keys
            m=ingestion
            n=scheduled_sweep
            o=backup
            p=new_documents_visible

            E:
            1:(a)-launched_with->(b)
            2:(c)-can_submit->(d)
            3:(e)-delayed_in->(f)
            4:(g)-not_root_cause_of->problem
            5:(h)-ran_on->(i)
            6:(h)-instead_of->(j)
            7:(k)-invalidate->(l)
            8:(k)-during->(m)
            9:(n)-serves_as->(o)
            10:(k)-restores->(p)
        """).strip(),
        "hypergraph": textwrap.dedent("""\
            N:
            a=search_feature_launch
            b=stale_cache
            c=problem
            d=users
            e=queries
            f=fresh_documents
            g=results
            h=delay
            i=ranking_model
            j=cache_invalidation
            k=delayed_schedule
            l=on_write
            m=fix
            n=affected_keys
            o=ingestion
            p=scheduled_sweep
            q=backup
            r=new_documents_visible

            H:
            H1:{a,b}-includes->{c}
            H2:{d}-can_submit->{e}
            H3:{f,g}-observed_as->{h}
            H4:{i}-not_root_cause_of->{c}
            H5:{j,k}-instead_of->{l}
            H6:{m}-invalidate->{n}
            H7:{m,o}-when->{n}
            H8:{p}-serves_as->{q}
            H9:{m}-restores->{r}
        """).strip(),
        "wenyan_hypergraph": textwrap.dedent("""\
            N:
            甲=检索新功
            乙=陈缓存
            丙=患
            丁=用户
            戊=发问
            己=新文
            庚=结果
            辛=迟
            壬=排序模型
            癸=缓存失效
            子=迟程
            丑=随写
            寅=解
            卯=受及诸键
            辰=摄入
            巳=定时清扫
            午=备
            未=新文速见

            H:
            一:{甲,乙}-并->{丙}
            二:{丁}-可->{戊}
            三:{己,庚}-见->{辛}
            四:{壬}-非本因->{丙}
            五:{癸,子}-替->{丑}
            六:{寅}-废->{卯}
            七:{寅,辰}-时->{卯}
            八:{巳}-为->{午}
            九:{寅}-复->{未}
        """).strip(),
    },
    {
        "id": "A4",
        "topic": "Approval bottleneck in governance flow",
        "english": (
            "A governance workflow slowed down because every tool call required manual approval. "
            "Critical actions were protected, but harmless read-only calls accumulated in the same queue. "
            "The policy engine was not broken. The real issue was that the approval rule lacked a distinction "
            "between write actions and safe reads. The fix was to keep approval for state-changing tools while "
            "allowing read-only tools to proceed automatically with audit logging. After that, response time improved "
            "without weakening control over risky operations."
        ),
        "wenyan": (
            "治理流程既缓，以诸工具调用皆须人审。危行固得护，然无害只读之调用，亦并积于同队。"
            "策略引擎非坏，实在审批之则未别写变与安全之读。其解：改状态者仍审，只读者则听其自动而行，"
            "并记审计。既而响应加速，而危操之制不弱。"
        ),
        "graph": textwrap.dedent("""\
            N:
            a=governance_workflow
            b=manual_approval_for_every_tool_call
            c=critical_actions
            d=read_only_calls
            e=same_queue
            f=policy_engine
            g=approval_rule
            h=write_actions
            i=safe_reads
            j=fix
            k=state_changing_tools
            l=automatic_read_only_tools
            m=audit_logging
            n=better_response_time
            o=risky_operations_control

            E:
            1:(a)-slowed_by->(b)
            2:(c)-status->protected
            3:(d)-accumulated_in->(e)
            4:(f)-not_broken->true
            5:(g)-lacked_distinction_between->(h)
            6:(g)-lacked_distinction_between->(i)
            7:(j)-keep_approval_for->(k)
            8:(j)-allow_automatic->(l)
            9:(j)-with->(m)
            10:(j)-improves->(n)
            11:(j)-preserves->(o)
        """).strip(),
        "hypergraph": textwrap.dedent("""\
            N:
            a=governance_workflow
            b=every_tool_call
            c=manual_approval
            d=slowdown
            e=critical_actions
            f=protected
            g=read_only_calls
            h=same_queue
            i=policy_engine
            j=approval_rule
            k=write_actions
            l=safe_reads
            m=fix
            n=state_changing_tools
            o=automatic_read_only_tools
            p=audit_logging
            q=better_response_time
            r=risky_operations_control

            H:
            H1:{a,b,c}-causes->{d}
            H2:{e}-status->{f}
            H3:{g}-accumulates_in->{h}
            H4:{i}-not_broken->{true}
            H5:{j}-lacks_distinction_between->{k,l}
            H6:{m}-keep_approval_for->{n}
            H7:{m}-allow_automatic->{o}
            H8:{m}-with->{p}
            H9:{m}-improves->{q}
            H10:{m}-preserves->{r}
        """).strip(),
        "wenyan_hypergraph": textwrap.dedent("""\
            N:
            甲=治理流程
            乙=诸调用
            丙=人审
            丁=缓
            戊=危行
            己=护
            庚=只读调用
            辛=同队
            壬=策略引擎
            癸=审批之则
            子=写变
            丑=安全之读
            寅=解
            卯=改状态工具
            辰=自动只读工具
            巳=审计记
            午=响应速
            未=危操之制

            H:
            一:{甲,乙,丙}-致->{丁}
            二:{戊}-状->{己}
            三:{庚}-积于->{辛}
            四:{壬}-非坏->{真}
            五:{癸}-未别->{子,丑}
            六:{寅}-仍审->{卯}
            七:{寅}-自动->{辰}
            八:{寅}-并->{巳}
            九:{寅}-益->{午}
            十:{寅}-守->{未}
        """).strip(),
    },
    {
        "id": "A5",
        "topic": "Notebook memory pressure",
        "english": (
            "A data notebook became unstable because a preprocessing step materialized several full copies of the same dataset. "
            "Charts still rendered, but later cells crashed with memory errors. The database was not at fault. The real issue was "
            "that each transformation returned a new in-memory frame instead of using views or chunked processing. The fix was to "
            "stream batches, reuse shared columns where possible, and persist intermediate outputs only when necessary. After that, "
            "the notebook remained responsive and peak memory dropped sharply."
        ),
        "wenyan": (
            "数据笔记本失稳，以预处理一步实化同集全副数本。图表犹可绘，而后诸格以存忆之错而崩。"
            "数据库非咎，实在每次变换皆返新帧于内存，不用视图与分块。其解：流其批次，可共列则复用之，"
            "中间产物惟必要时乃持久。既而笔记本仍敏，而峰值存忆骤降。"
        ),
        "graph": textwrap.dedent("""\
            N:
            a=data_notebook
            b=preprocessing_step
            c=full_copies_of_same_dataset
            d=charts
            e=later_cells
            f=memory_errors
            g=database
            h=transformations
            i=new_in_memory_frame
            j=views
            k=chunked_processing
            l=fix
            m=stream_batches
            n=reuse_shared_columns
            o=persist_intermediate_outputs_when_necessary
            p=responsive_notebook
            q=lower_peak_memory

            E:
            1:(a)-unstable_because_of->(b)
            2:(b)-materialized->(c)
            3:(d)-status->rendered
            4:(e)-crashed_with->(f)
            5:(g)-not_at_fault->true
            6:(h)-returned->(i)
            7:(h)-instead_of->(j)
            8:(h)-instead_of->(k)
            9:(l)-includes->(m)
            10:(l)-includes->(n)
            11:(l)-includes->(o)
            12:(l)-restores->(p)
            13:(l)-reduces->(q)
        """).strip(),
        "hypergraph": textwrap.dedent("""\
            N:
            a=data_notebook
            b=preprocessing_step
            c=full_copies_same_dataset
            d=instability
            e=charts
            f=rendered
            g=later_cells
            h=memory_errors
            i=database
            j=transformations
            k=new_in_memory_frame
            l=views
            m=chunked_processing
            n=fix
            o=stream_batches
            p=reuse_shared_columns
            q=persist_intermediate_outputs_when_necessary
            r=responsive_notebook
            s=lower_peak_memory

            H:
            H1:{a,b,c}-causes->{d}
            H2:{e}-status->{f}
            H3:{g}-crashes_with->{h}
            H4:{i}-not_at_fault->{true}
            H5:{j,k}-instead_of->{l,m}
            H6:{n}-includes->{o,p,q}
            H7:{n}-restores->{r}
            H8:{n}-reduces->{s}
        """).strip(),
        "wenyan_hypergraph": textwrap.dedent("""\
            N:
            甲=数据笔记本
            乙=预处理一步
            丙=同集多全副
            丁=失稳
            戊=图表
            己=可绘
            庚=后诸格
            辛=存忆错
            壬=数据库
            癸=诸变换
            子=新内存帧
            丑=视图
            寅=分块
            卯=解
            辰=流批次
            巳=复用共列
            午=必要时持久中间物
            未=仍敏
            申=峰值存忆降

            H:
            一:{甲,乙,丙}-致->{丁}
            二:{戊}-状->{己}
            三:{庚}-崩于->{辛}
            四:{壬}-非咎->{真}
            五:{癸,子}-替->{丑,寅}
            六:{卯}-兼->{辰,巳,午}
            七:{卯}-复->{未}
            八:{卯}-降->{申}
        """).strip(),
    },
]

encodings = {
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
}

rows = []
for sample in samples:
    base_ref = sample["english"]
    for variant in ["english", "wenyan", "graph", "hypergraph", "wenyan_hypergraph"]:
        text = sample[variant]
        row = {
            "article_id": sample["id"],
            "topic": sample["topic"],
            "representation": variant,
            "chars": len(text),
            "bytes_utf8": len(text.encode("utf-8")),
        }
        for enc_name, enc in encodings.items():
            tok = len(enc.encode(text))
            row[f"{enc_name}_tokens"] = tok
        rows.append(row)

df = pd.DataFrame(rows)

# savings vs English within each article
base = df[df["representation"] == "english"][["article_id", "cl100k_base_tokens", "o200k_base_tokens", "chars", "bytes_utf8"]].rename(
    columns={
        "cl100k_base_tokens": "base_cl100k",
        "o200k_base_tokens": "base_o200k",
        "chars": "base_chars",
        "bytes_utf8": "base_bytes",
    }
)
df = df.merge(base, on="article_id", how="left")
df["cl100k_saved"] = df["base_cl100k"] - df["cl100k_base_tokens"]
df["o200k_saved"] = df["base_o200k"] - df["o200k_base_tokens"]
df["cl100k_saved_pct"] = (df["cl100k_saved"] / df["base_cl100k"] * 100).round(1)
df["o200k_saved_pct"] = (df["o200k_saved"] / df["base_o200k"] * 100).round(1)

summary = (
    df.groupby("representation", as_index=False)[
        ["chars", "bytes_utf8", "cl100k_base_tokens", "o200k_base_tokens", "cl100k_saved", "o200k_saved"]
    ]
    .mean()
    .round(1)
    .sort_values("representation")
)

outdir = Path(".")/"data"/"hypergraph_wenyan_experiment"
outdir.mkdir(exist_ok=True)

# Save full dataset
csv_path = outdir / "token_experiment_results.csv"
df.to_csv(csv_path, index=False)

# Save code used to generate the experiment
script_path = outdir / "run_token_experiment.py"
# script_path.write_text(
#     "# Reproducible experiment code generated in ChatGPT sandbox\n"
#     "# Requires: pip install pandas tiktoken\n\n"
#     + open(__file__, "r").read() if "__file__" in globals() else
#     "# This notebook cell created the files. Copy the notebook code from the conversation if needed.\n",
#     encoding="utf-8"
# )

# Save a clean JSON corpus too
json_path = outdir / "sample_articles.json"
json_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

# Also save a compact markdown report
report_path = outdir / "README.md"
report_lines = [
    "# 文言文 + hypergraph token experiment",
    "",
    "Representations compared per article:",
    "- english",
    "- wenyan",
    "- graph",
    "- hypergraph",
    "- wenyan_hypergraph",
    "",
    "Tokenizers used:",
    "- cl100k_base",
    "- o200k_base",
    "",
    "Average tokens by representation:",
    "",
    # summary.to_markdown(index=False),
    "",
    "Per-article results are in `token_experiment_results.csv`.",
]
report_path.write_text("\n".join(report_lines), encoding="utf-8")

# display_dataframe_to_user(
#     "文言文 + hypergraph token experiment results",
    # df[[
    #     "article_id", "representation", "chars", "bytes_utf8",
    #     "cl100k_base_tokens", "o200k_base_tokens",
    #     "cl100k_saved", "cl100k_saved_pct",
    #     "o200k_saved", "o200k_saved_pct"
    # ]].sort_values(["article_id", "representation"])
# )
print(df)
print(f"Saved results to: {csv_path}")
print(f"Saved corpus to:   {json_path}")
print(f"Saved report to:   {report_path}")