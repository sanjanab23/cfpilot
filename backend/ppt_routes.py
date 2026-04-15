"""
ppt_routes.py  ── v2.0
Drop-in PPT route module for main.py.
No scheduling, no email — all other updates implemented.

UPGRADES:
  P1   /ppt/stream        — SSE streaming progress events
  P3   /ppt/download      — 25 MB PPTX size cap
  P4   /ppt/download      — slide JSON sanitized before generation
  S1   /ppt/download      — _sanitize applied to every slide dict
  S5   /ppt/plan + /ppt/download — audit logged
  F1   /ppt/regenerate    — regenerate a single slide (re-SOQL + re-insight)
  F2   /ppt/swap_chart    — swap chart type without re-querying
  F3   /ppt/variants      — executive / full / one-pager in parallel
  F4   /ppt/download      — watermark param (E4)
  F7   /ppt/pdf           — PDF export via LibreOffice headless
  F8   /ppt/reorder       — reorder slides

USAGE in main.py:
    from ppt_routes import register_ppt_routes
    register_ppt_routes(app, limiter, get_current_user_from_header,
                        get_user_context, audit_logger,
                        llm_rate_limiter, LLMInputSanitizer)
"""

import os
import re
import json
import time
import uuid
import logging
import tempfile
import subprocess
from typing import Optional, List

from fastapi import HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ppt_brain import generate_slide_plan
from ppt_generator import generate_pptx

logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
PPT_FILE_TTL  = 600            # 10 min
PPT_MAX_BYTES = 25_000_000     # P3 — 25 MB cap

# ─── IN-MEMORY FILE STORE ─────────────────────────────────────────────────────
_ppt_store: dict = {}

# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────

class PPTRequest(BaseModel):
    message:   str
    context:   Optional[str]  = None
    watermark: Optional[str]  = None   # E4: "DRAFT" | "CONFIDENTIAL"

class PPTEditRequest(BaseModel):
    slides:      list
    edit_index:  int
    edit_prompt: str

class PPTReorderRequest(BaseModel):
    slides:    list
    new_order: List[int]

class PPTSwapChartRequest(BaseModel):
    slides:         list
    slide_index:    int
    new_chart_type: str

class PPTRegenerateRequest(BaseModel):
    slides:      list
    slide_index: int
    user_query:  str

class PPTVariantsRequest(BaseModel):
    message:  str
    context:  Optional[str] = None

class PPTPDFRequest(BaseModel):
    slides: list

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _filename(slides: list) -> str:
    title = next((s.get("title","") for s in slides
                  if s.get("slide_type") == "cover" and s.get("title")), "")
    if not title:
        return "CFPilot_Presentation.pptx"
    safe = re.sub(r"[^\w\s\-]", "", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    safe = re.sub(r"_+", "_", safe)[:80]
    return f"{safe}.pptx" if safe else "CFPilot_Presentation.pptx"

def _sanitize(obj, depth: int = 0):
    """P4/S1 ── Sanitize slide JSON before use."""
    if depth > 5: return {}
    if isinstance(obj, dict):
        return {k: _sanitize(v, depth+1) for k, v in obj.items()
                if isinstance(k, str) and len(k) <= 100}
    if isinstance(obj, list):
        return [_sanitize(i, depth+1) for i in obj[:200]]
    if isinstance(obj, str):
        return obj[:500]
    return obj

def _store(pptx_bytes: bytes, slides: list, owner: str) -> dict:
    fname = _filename(slides)
    token = str(uuid.uuid4())
    path  = os.path.join(tempfile.gettempdir(), f"cfpilot_{token}.pptx")
    with open(path, "wb") as f:
        f.write(pptx_bytes)
    _ppt_store[token] = {"path": path, "owner": owner,
                          "created_at": time.time(), "filename": fname}
    return {"token": token, "path": path, "filename": fname}

def _cleanup():
    now   = time.time()
    stale = [t for t, e in list(_ppt_store.items())
             if now - e.get("created_at", 0) > PPT_FILE_TTL]
    for t in stale:
        e = _ppt_store.pop(t, None)
        if e:
            try: os.unlink(e["path"])
            except OSError: pass
    if stale:
        logger.info(f"Cleaned {len(stale)} stale PPT file(s)")

async def _sse(data: str) -> str:
    return f"data: {data}\n\n"

VALID_CHART_TYPES = {
    "bar","line","pie","donut","waterfall","bullet","treemap","scatter","combo"
}

# ─── ROUTE REGISTRATION ───────────────────────────────────────────────────────

def register_ppt_routes(app, limiter, get_current_user_from_header,
                        get_user_context, audit_logger,
                        llm_rate_limiter, LLMInputSanitizer):
    """
    Register all PPT routes onto the FastAPI app.

    Call once in main.py after all middleware is configured:
        register_ppt_routes(app, limiter, get_current_user_from_header,
                            get_user_context, audit_logger,
                            llm_rate_limiter, LLMInputSanitizer)
    """

    # ── /ppt/plan ─────────────────────────────────────────────────────────────
    @app.post("/ppt/plan")
    @limiter.limit("10/minute")
    async def ppt_plan(
        request: Request,
        data: PPTRequest,
        background_tasks: BackgroundTasks,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """Generate slide plan JSON. S5 — audit logged."""
        if not llm_rate_limiter.check_limit(user_email):
            raise HTTPException(429, "Rate limit exceeded")
        try:
            query = LLMInputSanitizer.sanitize(data.message)
        except ValueError:
            raise HTTPException(400, "Invalid input")

        ctx = get_user_context(user_email)
        logger.info(f"PPT plan: {user_email} | {query[:80]}")
        try:
            slides = generate_slide_plan(query, ctx)
            # S5
            if hasattr(audit_logger, "log_action"):
                audit_logger.log_action(user_email, "ppt_plan",
                                         f"slides={len(slides)} q={query[:60]}")
            background_tasks.add_task(_cleanup)
            return {"status": "success", "slides": slides, "count": len(slides)}
        except Exception as e:
            logger.error(f"ppt_plan error {user_email}: {e}")
            raise HTTPException(500, "Failed to generate plan")

    # ── /ppt/stream ───────────────────────────────────────────────────────────
    @app.get("/ppt/stream")
    @limiter.limit("10/minute")
    async def ppt_stream(
        request: Request,
        message: str,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """P1 ── SSE streaming progress. Connect with EventSource."""
        if not llm_rate_limiter.check_limit(user_email):
            raise HTTPException(429, "Rate limit exceeded")
        try:
            query = LLMInputSanitizer.sanitize(message)
        except ValueError:
            raise HTTPException(400, "Invalid input")

        ctx = get_user_context(user_email)

        async def _events():
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            yield await _sse(json.dumps({"status": "starting",
                                          "message": "Initialising PPT pipeline..."}))
            result_box = {}

            def _run():
                result_box["slides"] = generate_slide_plan(query, ctx)

            stages = [
                ("querying",   "Running Salesforce queries..."),
                ("analysing",  "Detecting anomalies in data..."),
                ("generating", "Generating slide insights..."),
                ("building",   "Building presentation..."),
            ]

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run)
                for sk, sm in stages:
                    if fut.done():
                        break
                    yield await _sse(json.dumps({"status": sk, "message": sm}))
                    await asyncio.sleep(4)
                try:
                    fut.result(timeout=90)
                except Exception as e:
                    yield await _sse(json.dumps({"status": "error", "message": str(e)}))
                    return

            slides = result_box.get("slides", [])
            yield await _sse(json.dumps({
                "status":  "complete",
                "message": f"{len(slides)} slides generated",
                "slides":  slides,
                "count":   len(slides),
            }))

        return StreamingResponse(_events(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})

    # ── /ppt/edit ─────────────────────────────────────────────────────────────
    @app.post("/ppt/edit")
    @limiter.limit("20/minute")
    async def ppt_edit(
        request: Request,
        data: PPTEditRequest,
        user_email: str = Depends(get_current_user_from_header)
    ):
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage as SM

        try:
            instr = LLMInputSanitizer.sanitize(data.edit_prompt)
        except ValueError:
            raise HTTPException(400, "Invalid edit instruction")

        slides = data.slides
        idx    = data.edit_index
        if not (0 <= idx < len(slides)):
            raise HTTPException(400, "Invalid slide index")

        slide_safe = _sanitize(slides[idx])  # P4/S1

        llm_e = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="xiaomi/mimo-v2-flash", temperature=0.2, max_retries=3,
            model_kwargs={"extra_body": {"provider": {"zdr": True}}}
        )
        prompt = (f"Edit this slide JSON per the instruction. "
                  f"Return JSON ONLY — no markdown.\n\n"
                  f"Slide:\n{json.dumps(slide_safe, indent=2)}\n\n"
                  f"Instruction: {instr}\n\nUpdated slide JSON:")
        try:
            resp  = llm_e.invoke([SM(content=prompt)])
            clean = re.sub(r"^```(?:json)?\s*","",resp.content.strip())
            clean = re.sub(r"\s*```$","",clean).strip()
            slides[idx] = json.loads(clean)
            return {"status": "success", "slides": slides}
        except Exception as e:
            logger.error(f"ppt_edit error: {e}")
            raise HTTPException(500, "Failed to apply edit")

    # ── /ppt/reorder ─────────────────────────────────────────────────────────
    @app.post("/ppt/reorder")
    @limiter.limit("30/minute")
    async def ppt_reorder(
        request: Request,
        data: PPTReorderRequest,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """F8 ── Reorder slides by supplying permutation of original indices."""
        s, o = data.slides, data.new_order
        if len(o) != len(s):
            raise HTTPException(400, "new_order length must match slides length")
        if sorted(o) != list(range(len(s))):
            raise HTTPException(400, "new_order must be a valid permutation")
        return {"status": "success", "slides": [s[i] for i in o]}

    # ── /ppt/swap_chart ───────────────────────────────────────────────────────
    @app.post("/ppt/swap_chart")
    @limiter.limit("30/minute")
    async def ppt_swap_chart(
        request: Request,
        data: PPTSwapChartRequest,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """F2 ── Swap chart type for one slide without re-querying Salesforce."""
        if data.new_chart_type not in VALID_CHART_TYPES:
            raise HTTPException(400, f"chart_type must be one of {VALID_CHART_TYPES}")
        slides = data.slides
        idx    = data.slide_index
        if not (0 <= idx < len(slides)):
            raise HTTPException(400, "Invalid slide index")
        slides[idx]["chart_type"] = data.new_chart_type
        return {"status": "success", "slides": slides,
                "message": f"Slide {idx} → '{data.new_chart_type}'"}

    # ── /ppt/regenerate ───────────────────────────────────────────────────────
    @app.post("/ppt/regenerate")
    @limiter.limit("10/minute")
    async def ppt_regenerate(
        request: Request,
        data: PPTRegenerateRequest,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """F1 ── Re-run SOQL + insight for a single slide and splice it back in."""
        if not llm_rate_limiter.check_limit(user_email):
            raise HTTPException(429, "Rate limit exceeded")
        try:
            query = LLMInputSanitizer.sanitize(data.user_query)
        except ValueError:
            raise HTTPException(400, "Invalid input")

        slides = data.slides
        idx    = data.slide_index
        if not (0 <= idx < len(slides)):
            raise HTTPException(400, "Invalid slide index")

        ctx   = get_user_context(user_email)
        fresh = generate_slide_plan(f"Generate ONE slide for: {query}", ctx)
        if not fresh:
            raise HTTPException(500, "Failed to regenerate slide")

        target_type = slides[idx].get("slide_type", "chart")
        replacement = next((s for s in fresh if s.get("slide_type") == target_type),
                           fresh[0])
        slides[idx] = replacement
        logger.info(f"Slide {idx} regenerated for {user_email}")
        return {"status": "success", "slides": slides}

    # ── /ppt/variants ─────────────────────────────────────────────────────────
    @app.post("/ppt/variants")
    @limiter.limit("5/minute")
    async def ppt_variants(
        request: Request,
        data: PPTVariantsRequest,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """
        F3 ── Generate 3 variants in parallel:
          executive (5 slides) / full (12-15 slides) / onepager (1 content slide)
        """
        if not llm_rate_limiter.check_limit(user_email):
            raise HTTPException(429, "Rate limit exceeded")
        try:
            query = LLMInputSanitizer.sanitize(data.message)
        except ValueError:
            raise HTTPException(400, "Invalid input")

        ctx = get_user_context(user_email)
        variant_qs = {
            "executive": f"Concise executive summary deck (max 5 slides, big KPI numbers): {query}",
            "full":      query,
            "onepager":  f"Single-slide one-pager with top 5 key metrics: {query}",
        }

        from concurrent.futures import ThreadPoolExecutor, as_completed as asc
        results = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            fmap = {ex.submit(generate_slide_plan, vq, ctx): vn
                    for vn, vq in variant_qs.items()}
            for fut in asc(fmap, timeout=120):
                vn = fmap[fut]
                try:
                    results[vn] = fut.result()
                except Exception as e:
                    logger.error(f"Variant '{vn}' failed: {e}")
                    results[vn] = []

        return {"status":   "success",
                "variants": results,
                "counts":   {k: len(v) for k, v in results.items()}}

    # ── /ppt/download ─────────────────────────────────────────────────────────
    @app.post("/ppt/download")
    @limiter.limit("10/minute")
    async def ppt_download(
        request: Request,
        data: dict,
        background_tasks: BackgroundTasks,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """
        P3 — 25 MB size cap
        P4/S1 — slide JSON sanitized
        E4 — watermark support
        S5 — audit logged
        """
        slides    = data.get("slides", [])
        watermark = data.get("watermark", None)  # E4
        if not slides:
            raise HTTPException(400, "No slides provided")

        # P4/S1
        slides = [_sanitize(s) for s in slides]

        try:
            pptx = generate_pptx(slides, watermark=watermark)
        except Exception as e:
            logger.error(f"generate_pptx error {user_email}: {e}")
            raise HTTPException(500, "Failed to generate PPTX")

        # P3
        if len(pptx) > PPT_MAX_BYTES:
            raise HTTPException(
                413,
                f"PPTX too large ({len(pptx)/1e6:.1f} MB > 25 MB). "
                "Reduce slide count or data density."
            )

        info = _store(pptx, slides, user_email)
        background_tasks.add_task(_cleanup)

        # S5
        if hasattr(audit_logger, "log_action"):
            audit_logger.log_action(user_email, "ppt_download",
                                     f"slides={len(slides)} size={len(pptx):,}B "
                                     f"file={info['filename']}")

        logger.info(f"PPTX {user_email}: {len(pptx):,}B | {info['filename']}")
        return {"status":       "success",
                "download_url": f"/ppt/file/{info['token']}",
                "filename":     info["filename"],
                "size_bytes":   len(pptx)}

    # ── /ppt/file/{token} ─────────────────────────────────────────────────────
    @app.get("/ppt/file/{token}")
    async def ppt_file(token: str, background_tasks: BackgroundTasks):
        entry = _ppt_store.get(token)
        if not entry or not os.path.exists(entry["path"]):
            _ppt_store.pop(token, None)
            raise HTTPException(404, "File not found or expired")

        def _del():
            _ppt_store.pop(token, None)
            try: os.unlink(entry["path"])
            except OSError: pass

        background_tasks.add_task(_del)
        return FileResponse(
            entry["path"],
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=entry.get("filename", "presentation.pptx"),
        )

    # ── /ppt/pdf ─────────────────────────────────────────────────────────────
    @app.post("/ppt/pdf")
    @limiter.limit("5/minute")
    async def ppt_pdf(
        request: Request,
        data: PPTPDFRequest,
        background_tasks: BackgroundTasks,
        user_email: str = Depends(get_current_user_from_header)
    ):
        """F7 ── Export slides as PDF via LibreOffice headless."""
        slides = [_sanitize(s) for s in data.slides]
        if not slides:
            raise HTTPException(400, "No slides provided")

        try:
            pptx_bytes = generate_pptx(slides)
        except Exception as e:
            raise HTTPException(500, f"PPTX generation failed: {e}")

        token     = str(uuid.uuid4())
        tmp_pptx  = os.path.join(tempfile.gettempdir(), f"cfp_{token}.pptx")
        tmp_pdf   = tmp_pptx.replace(".pptx", ".pdf")

        try:
            with open(tmp_pptx, "wb") as f:
                f.write(pptx_bytes)

            # Try soffice wrapper first, then direct soffice
            cmds = [
                ["python3", "scripts/office/soffice.py",
                 "--headless", "--convert-to", "pdf",
                 "--outdir", tempfile.gettempdir(), tmp_pptx],
                ["soffice", "--headless", "--convert-to", "pdf",
                 "--outdir", tempfile.gettempdir(), tmp_pptx],
            ]
            converted = False
            for cmd in cmds:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if r.returncode == 0 and os.path.exists(tmp_pdf):
                    converted = True
                    break

            if not converted:
                raise RuntimeError("LibreOffice conversion failed")

            with open(tmp_pdf, "rb") as f:
                pdf_bytes = f.read()

        except subprocess.TimeoutExpired:
            raise HTTPException(504, "PDF conversion timed out (>60s)")
        except Exception as e:
            logger.error(f"PDF conversion error: {e}")
            raise HTTPException(500, f"PDF conversion failed: {e}")
        finally:
            for p in (tmp_pptx, tmp_pdf):
                try: os.unlink(p)
                except OSError: pass

        # Store PDF for download
        pdf_token = str(uuid.uuid4())
        pdf_path  = os.path.join(tempfile.gettempdir(), f"cfp_pdf_{pdf_token}.pdf")
        pdf_fname = _filename(slides).replace(".pptx", ".pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        _ppt_store[pdf_token] = {"path": pdf_path, "owner": user_email,
                                   "created_at": time.time(), "filename": pdf_fname}
        background_tasks.add_task(_cleanup)

        return {"status":       "success",
                "download_url": f"/ppt/file/{pdf_token}",
                "filename":     pdf_fname,
                "size_bytes":   len(pdf_bytes)}