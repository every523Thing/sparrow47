from __future__ import annotations

import difflib
import hashlib
import io
import json
import random
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Unpack, cast

from PIL import Image
from openai import APIConnectionError, APIError, APITimeoutError
from openai.types.responses import Response, ResponseInputParam
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from witness_client import ExchangePending, RenderOptions, RenderRejected, RenderUnavailable, WitnessError

from . import patch, prompts
from .budget import CREDIT_LIMIT, PRICING, SAFETY, Budget, worst_case
from .fallback import fallback_module
from .images import data_url, object_closeup, png_data_url, to_jpeg, view_sheet, zoomed_views
from .orient import Renderer

SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"
SELECT_EFFORT = "medium"
GENERATE_MAX_OUTPUT = 32768
SELECT_MAX_OUTPUT = 8192
GARBAGE_PENALTY = 2.0
ASTRA_MAX_OUTPUT = 22000
EFFORT_MAX_OUTPUT = {"xhigh": 44000, "max": 64000}
GRID_WEIGHT = 0.25
PARENT_MARGIN = 0.25
JOIN_ATTEMPTS = 6
RENDER_WAITS = (5, 10, 20, 40, 60, 60)
RENDER_TIMEOUT_RETRIES = 2
CHAIN_ROOM = 24
DRAFT_ID = 1
REDRAFT_ID = 50
REWRITE_BASE = 100
REPAIR_OFFSET = 5000
SELECT_ID = 9000
FINAL_SELECT_ID = 9800
TEXT_READ_ID = 9985  # TR: the text read-out before the drafts
CHECK_ID = 9950
CRITIQUE_ID = 9400
JUDGE_MAX_OUTPUT = 4096
JUDGE_DRAW = 0.7
CHECK_SEVERITY = 5

FRONT_JUDGE = RenderOptions(thetas=(30, 330, 0, 0), phis=(0, 0, 15, -30), img_size=1024, gap=0, bg_color="ffffff",
                            lighting="neutral")
GRID_VIEW = RenderOptions(lighting="neutral")

WRAPPER = """
function __drawCalls404(root) {
  let count = 0;
  root.traverse(function (node) { if (node.isMesh || node.isLine || node.isPoints) count += 1; });
  return count;
}

function __mergeByMaterial404(THREE, root) {
  root.updateMatrixWorld(true);
  const byMaterial = new Map();
  const groups = [];
  root.traverse(function (node) {
    if (!node.isMesh || node.isInstancedMesh || !node.visible || node.children.length || Array.isArray(node.material)) return;
    const geometry = node.geometry;
    if (!geometry || !geometry.attributes.position) return;
    if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length) return;
    const names = Object.keys(geometry.attributes).sort();
    const key = names.map(function (n) { return n + ':' + geometry.attributes[n].itemSize; }).join(',');
    if (!byMaterial.has(node.material)) byMaterial.set(node.material, new Map());
    const byLayout = byMaterial.get(node.material);
    if (!byLayout.has(key)) { const group = { names: names, meshes: [] }; byLayout.set(key, group); groups.push(group); }
    byLayout.get(key).meshes.push(node);
  });
  const vector = new THREE.Vector3();
  const normalMatrix = new THREE.Matrix3();
  const merged = [];
  for (const group of groups) {
    const meshes = group.meshes;
    if (meshes.length < 2) continue;
    let vertexCount = 0;
    let indexCount = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      vertexCount += geometry.attributes.position.count;
      indexCount += geometry.index ? geometry.index.count : geometry.attributes.position.count;
    }
    const arrays = {};
    for (const name of group.names) arrays[name] = new Float32Array(vertexCount * meshes[0].geometry.attributes[name].itemSize);
    const index = vertexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
    let vertexOffset = 0;
    let indexOffset = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      const matrix = mesh.matrixWorld;
      normalMatrix.getNormalMatrix(matrix);
      const count = geometry.attributes.position.count;
      for (const name of group.names) {
        const attribute = geometry.attributes[name];
        const size = attribute.itemSize;
        const out = arrays[name];
        for (let i = 0; i < count; i++) {
          const base = (vertexOffset + i) * size;
          if (name === 'position' || name === 'normal') {
            vector.fromBufferAttribute(attribute, i);
            if (name === 'position') vector.applyMatrix4(matrix); else vector.applyNormalMatrix(normalMatrix);
            out[base] = vector.x; out[base + 1] = vector.y; out[base + 2] = vector.z;
          } else {
            out[base] = attribute.getX(i);
            if (size > 1) out[base + 1] = attribute.getY(i);
            if (size > 2) out[base + 2] = attribute.getZ(i);
            if (size > 3) out[base + 3] = attribute.getW(i);
          }
        }
      }
      const flip = matrix.determinant() < 0;
      const triangles = geometry.index ? geometry.index.count : count;
      for (let i = 0; i < triangles; i++) {
        const source = flip ? i - (i % 3) + [0, 2, 1][i % 3] : i;
        index[indexOffset + i] = (geometry.index ? geometry.index.getX(source) : source) + vertexOffset;
      }
      vertexOffset += count;
      indexOffset += triangles;
    }
    const combined = new THREE.BufferGeometry();
    for (const name of group.names) {
      combined.setAttribute(name, new THREE.BufferAttribute(arrays[name], meshes[0].geometry.attributes[name].itemSize));
    }
    combined.setIndex(new THREE.BufferAttribute(index, 1));
    for (const mesh of meshes) mesh.parent.remove(mesh);
    merged.push(new THREE.Mesh(combined, meshes[0].material));
  }
  for (const mesh of merged) root.add(mesh);
}

function __textures404(root) {
  const found = new Map();
  root.traverse(function (node) {
    const materials = node.material ? [].concat(node.material) : [];
    for (const material of materials) {
      for (const key of Object.keys(material)) {
        const value = material[key];
        if (value && value.isDataTexture && value.image && value.image.data) {
          if (!found.has(value)) found.set(value, []);
          found.get(value).push([material, key]);
        }
      }
    }
  });
  return found;
}

function __shrinkTextures404(THREE, root, limit) {
  for (let pass = 0; pass < 8; pass++) {
    const found = __textures404(root);
    let total = 0;
    let largest = null;
    for (const texture of found.keys()) {
      total += texture.image.data.byteLength;
      if (!largest || texture.image.data.byteLength > largest.image.data.byteLength) largest = texture;
    }
    if (total <= limit || !largest || largest.image.width < 16 || largest.image.height < 16) return;
    const width = largest.image.width;
    const height = largest.image.height;
    const data = largest.image.data;
    const channels = Math.round(data.length / (width * height));
    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);
    const size = halfWidth * halfHeight * channels;
    const out = data instanceof Float32Array ? new Float32Array(size)
      : data instanceof Uint16Array ? new Uint16Array(size) : new Uint8Array(size);
    for (let y = 0; y < halfHeight; y++) {
      for (let x = 0; x < halfWidth; x++) {
        for (let c = 0; c < channels; c++) {
          const a = ((2 * y) * width + 2 * x) * channels + c;
          const b = a + channels;
          const d = a + width * channels;
          out[(y * halfWidth + x) * channels + c] = (data[a] + data[b] + data[d] + data[d + channels]) / 4;
        }
      }
    }
    const smaller = new THREE.DataTexture(out, halfWidth, halfHeight, largest.format, largest.type);
    smaller.wrapS = largest.wrapS;
    smaller.wrapT = largest.wrapT;
    smaller.magFilter = largest.magFilter;
    smaller.minFilter = largest.minFilter;
    smaller.generateMipmaps = largest.generateMipmaps;
    smaller.colorSpace = largest.colorSpace;
    smaller.flipY = largest.flipY;
    smaller.repeat.copy(largest.repeat);
    smaller.offset.copy(largest.offset);
    smaller.needsUpdate = true;
    for (const [material, key] of found.get(largest)) material[key] = smaller;
  }
}

export default function generate(THREE) {
  const root = new THREE.Group();
  root.add(__asset404(THREE));
  if (__drawCalls404(root) > 200) __mergeByMaterial404(THREE, root);
  __shrinkTextures404(THREE, root, 4100000);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (extent > 0 && Number.isFinite(extent)) {
    const center = box.getCenter(new THREE.Vector3());
    const scale = 0.95 / extent;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
  return root;
}
"""
_FIT = "  const box = new THREE.Box3().setFromObject(root);\n"


_EXPORT_LINE = "export default function generate(THREE) {\n"
_OUTLINE_FUNCTION = """
function __bake404(THREE, root) {
  root.updateMatrixWorld(true);
  let vertices = 0;
  root.traverse(function (node) {
    if (node.geometry && node.geometry.attributes.position && !node.isInstancedMesh) vertices += node.geometry.attributes.position.count;
  });
  if (vertices > 200000) return;
  const instance = new THREE.Matrix4();
  const offset = new THREE.Matrix4();
  const places = new Map();
  const reversed = function (geometry) {
    const count = geometry.index ? geometry.index.count : geometry.attributes.position.count;
    const order = new Uint32Array(count - (count % 3));
    for (let i = 0; i < order.length; i++) {
      const corner = i - (i % 3) + [0, 2, 1][i % 3];
      order[i] = geometry.index ? geometry.index.getX(corner) : corner;
    }
    geometry.setIndex(new THREE.BufferAttribute(order, 1));
    return geometry;
  };
  root.traverse(function (node) {
    const isShape = node.isMesh || node.isLine || node.isPoints;
    if (!isShape || !node.geometry || !node.geometry.attributes.position) return;
    const place = new THREE.Vector3().setFromMatrixPosition(node.matrixWorld);
    places.set(node, place);
    offset.makeTranslation(-place.x, -place.y, -place.z);
    if (node.isInstancedMesh) {
      for (let i = 0; i < node.count; i++) {
        node.getMatrixAt(i, instance);
        node.setMatrixAt(i, instance.premultiply(node.matrixWorld).premultiply(offset));
      }
      node.instanceMatrix.needsUpdate = true;
      node.boundingBox = null;
      node.boundingSphere = null;
      if (node.matrixWorld.determinant() < 0) node.geometry = reversed(node.geometry.clone());
      return;
    }
    let geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    geometry.applyMatrix4(offset);
    if (node.isMesh && node.matrixWorld.determinant() < 0) geometry = reversed(geometry);
    node.geometry = geometry;
  });
  root.traverse(function (node) {
    const place = places.get(node);
    let up = new THREE.Vector3();
    for (let p = node.parent; p && p !== root.parent; p = p.parent) {
      const q = places.get(p);
      if (q) { up = q; break; }
    }
    if (place) node.position.copy(place).sub(up); else node.position.set(0, 0, 0);
    node.quaternion.identity();
    node.scale.set(1, 1, 1);
  });
  root.updateMatrixWorld(true);
}
"""
_BAKE_RESET = "    node.scale.set(1, 1, 1);\n  });\n  root.updateMatrixWorld(true);\n}\n"
_BAKE_RESET_FIXED = "    node.scale.set(1, 1, 1);\n    node.updateMatrix();\n  });\n  root.updateMatrixWorld(true);\n}\n"


def chain_wrapper(fill: float, outline_fit: bool = False) -> str:
    wrapper = WRAPPER.replace("0.95 / extent", f"{fill} / extent", 1)
    if outline_fit:
        bake = _OUTLINE_FUNCTION.replace(_BAKE_RESET, _BAKE_RESET_FIXED, 1)
        wrapper = wrapper.replace(_FIT, "  __bake404(THREE, root);\n" + _FIT, 1).replace(_EXPORT_LINE, bake + "\n" + _EXPORT_LINE, 1)
    return wrapper


_EXPORT = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\(")


class ModelCalls(Protocol):
    def create(self, *, request_id: int = 0, **params: Unpack[ResponseCreateParamsBase]) -> Response: ...


class PatientRenderer:
    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        self.gateway_codes: tuple[int, ...] = (500, 502, 504)
        self.cap = 0
        self.room = 0
        self.rendered: set[str] = set()
        self.refused = 0

    def render(self, source: str, options: RenderOptions, room: int = 0) -> bytes:
        return self._counted(source, options, room)

    def over(self, room: int) -> bool:
        return self.cap > 0 and len(self.rendered) >= self.cap - room

    @contextmanager
    def reserving(self, room: int) -> Iterator[None]:
        saved = self.room
        self.room = max(saved, room)
        try:
            yield
        finally:
            self.room = saved

    def _counted(self, source: str, options: RenderOptions, room: int) -> bytes:
        key = hashlib.sha256(json.dumps([options.query(), source], sort_keys=True).encode()).hexdigest()
        if key not in self.rendered and self.over(max(room, self.room)):
            self.refused += 1
            raise _render_capped()
        image = self._patient(source, options)
        self.rendered.add(key)
        return image

    def _patient(self, source: str, options: RenderOptions) -> bytes:
        waits = iter(RENDER_WAITS)
        timeouts = 0
        busy = 2
        while True:
            try:
                return self._renderer.render(source, options)
            except RenderRejected as rejection:
                timeouts += 1
                if "TIMEOUT_EXCEEDED" not in rejection.detail_json or timeouts > RENDER_TIMEOUT_RETRIES:
                    raise
            except RenderUnavailable as error:
                response = getattr(error.__cause__, "response", None)
                if getattr(response, "status_code", None) in self.gateway_codes:
                    timeouts += 1
                    if timeouts > RENDER_TIMEOUT_RETRIES:
                        raise _render_timeout() from error
                if getattr(response, "status_code", None) == 503:
                    time.sleep(busy)
                    busy = min(busy * 2, 10)
                    continue
                if getattr(response, "status_code", None) == 429:
                    raise
                wait = next(waits, None)
                if wait is None:
                    raise
                time.sleep(wait)
            except (ExchangePending, WitnessError):
                raise
            except Exception as error:  # a gateway that gave up on a slow preview render (HTTP 500, 502 or 504 from the Pod)
                if getattr(getattr(error, "response", None), "status_code", None) not in self.gateway_codes:
                    raise
                timeouts += 1
                if timeouts > RENDER_TIMEOUT_RETRIES:
                    raise _render_timeout() from error
                time.sleep(next(waits, 60))


def _render_capped() -> RenderRejected:
    return RenderRejected(json.dumps({"error": "RENDER_CAP: the per-image preview render allowance is spent."}))


def _render_timeout() -> RenderRejected:
    detail = "RENDER_TIMEOUT: the renderer could not finish this module in time. Make it lighter: "
    return RenderRejected(json.dumps({"error": detail + "fewer meshes, fewer vertices, lower segment counts."}))


OVERLOAD_WAITS = (5, 15, 30, 60)
OVERLOADED = ("Unusable provider response", "currently overloaded", "server_is_overloaded", "HTTP 429")
CREDENTIAL_SIGNS = ("credential", "subscription")


class CallFailed(Exception):
    pass


@dataclass
class Candidate:
    name: str
    source: str
    front: bytes
    sides: bytes
    raw: str = ""
    views: list[bytes] = field(default_factory=list)
    parent: Candidate | None = None
    fixes: list[str] = field(default_factory=list)
    notes: str = ""
    shots: list[bytes] = field(default_factory=list)
    uneven: bool = False  # the model's module scales the whole object unevenly (see uneven_scale)


@dataclass
class Judgement:
    score: float
    fixes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Draft:
    start: str
    model: str
    effort: str


@dataclass(frozen=True)
class Step:
    rewrites: tuple[tuple[str, str], ...] = ()
    focus: tuple[str, ...] = ()
    fallback: tuple[str, str] | None = None
    patch: bool = False
    info: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagePlan:
    drafts: tuple[Draft, ...]
    steps: tuple[Step, ...]
    draft_keep: int = 0
    fill: float = 0.99
    keep_dark: bool = False
    merge_call: tuple[str, str] = (SOL, "high")
    merge_others: int = 2
    merge_rounds: int = 1
    render_cap: int = 0


@dataclass
class Call:
    request_id: int
    model: str
    instructions: str
    content: list[dict[str, Any]]
    effort: str
    max_output_tokens: int
    text: dict[str, Any] | None = None
    estimate: float = 0.0


def extract_source(text: str) -> str | None:
    text = text.strip()
    blocks = [b for b in re.findall(r"```(?:javascript|js|mjs)?[^\n]*\n([\s\S]*?)```", text) if _EXPORT.search(b)]
    if blocks:
        text = blocks[-1].strip()
    return text if _EXPORT.search(text) else None


def normalize(source: str, wrapper: str = WRAPPER) -> str:
    match = _EXPORT.search(source)
    if match is None:
        return source
    inner = source[: match.start()] + "function __asset404(" + source[match.end():]
    return inner + "\n" + wrapper


class Pipeline:
    def __init__(self, image: bytes, seed: int, calls: ModelCalls, renderer: Renderer) -> None:
        self.image = image
        self.seed = seed
        self.history: list[tuple[int, str, bool]] = []
        self.changes: list[tuple[int, str, bool]] = []
        self.calls = calls
        self.renderer = PatientRenderer(renderer)
        self.budget = Budget()
        self.target = {"type": "input_image", "image_url": png_data_url(image), "detail": "high"}
        self.digest = hashlib.sha256(image).hexdigest()
        self.best_so_far: str | None = None
        self.wrapper = WRAPPER
        self.closeup: list[dict[str, Any]] = []
        self.instructions = prompts.GENERATE_CHAIN
        self.astra_cap = ASTRA_MAX_OUTPUT
        self.patch_ratios: list[float] = []
        self.end_reserve = 0.0
        self.gate_notes = ""
        self.plan = StagePlan(drafts=(), steps=())
        self.salt = 0

    def wrap(self, raw: str, wrapper: str | None = None) -> str:
        return normalize(raw, self.wrapper if wrapper is None else wrapper)

    def merge_call(self, request_id: int, parent: Candidate, child: Candidate) -> Call:
        image = lambda b: {"type": "input_image", "image_url": data_url(b), "detail": "high"}
        return self.module_call(request_id, SOL, "high", [
            {"type": "input_text", "text": "CURRENT version: close-ups from the judge's four front cameras (30 degrees left, 30 degrees right, from slightly below, from above), then its grid:"},
            *[image(v) for v in parent.views], image(parent.sides),
            {"type": "input_text", "text": "NEW version (its latest rewrite): the same views:"},
            *[image(v) for v in child.views], image(child.sides),
            {"type": "input_text", "text": prompts.MERGE_ROUND.format(current=parent.raw, new=child.raw)}])

    def checked(self, pool: ThreadPoolExecutor, winner: Candidate) -> Candidate:
        if not winner.raw or not winner.views:
            return winner
        views = [{"type": "input_image", "image_url": data_url(view), "detail": "high"} for view in winner.views]
        call = Call(CHECK_ID, SOL, prompts.CHECK_SYSTEM, [self.target, *views, {"type": "input_text", "text": prompts.CHECK_USER}],
                    SELECT_EFFORT, SELECT_MAX_OUTPUT, text=prompts.check_schema())
        reply = self.run_stage(pool, [call])[0]
        try:
            found = json.loads(reply.output_text) if reply is not None else {}
            faults, severity = [str(f) for f in found.get("faults", [])][:6], int(found.get("severity", 0))
        except (ValueError, TypeError, AttributeError):
            return winner
        if not faults or severity < CHECK_SEVERITY:
            return winner
        note = prompts.CHECK_NOTE.format(faults="\n".join(f"- {fault}" for fault in faults))
        child = self.modules(pool, [self.rewrite_call(CHECK_ID + 10, SOL, "high", winner, "", note)], ["checked"])[0]
        if child is None:
            return winner
        return self.select(pool, [child, winner], {winner.name}, 1, CHECK_ID + 20)[0]

    def merged_pick(self, pool: ThreadPoolExecutor, drafts: list[Candidate], picked: Candidate, offset: int = 0) -> Candidate:
        if len(drafts) < 2:
            return picked
        labels = [chr(65 + i) for i in range(len(drafts))]
        by = dict(zip(labels, drafts, strict=True))
        image = lambda b: {"type": "input_image", "image_url": data_url(b), "detail": "high"}
        content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup]
        for label, draft in by.items():
            content += [{"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"}, image(draft.front),
                        {"type": "input_text", "text": f"CANDIDATE {label} GRID:"}, image(draft.sides)]
        content.append({"type": "input_text", "text": prompts.CRITIQUE_USER})
        reply = self.run_stage(pool, [Call(CRITIQUE_ID + offset, SOL, prompts.CRITIQUE_SYSTEM, content, "medium", SELECT_MAX_OUTPUT,
                                           text=prompts.critique_schema(labels))])[0]
        try:
            critique = json.loads(reply.output_text) if reply is not None else None
            notes = {entry["label"]: entry for entry in critique["candidates"] if entry.get("label") in by}
            base = critique["base"] if critique.get("base") in by else labels[drafts.index(picked)]
        except (ValueError, TypeError, KeyError, AttributeError):
            return picked
        others = sorted((label for label in labels if label != base),
                        key=lambda label: -len(notes.get(label, {}).get("strengths", [])))[:self.plan.merge_others]
        notes.update(self.draft_notes_each(pool, by, [base, *others], offset))

        def block(label: str, role: str) -> str:
            note = notes.get(label, {})
            return (f"{role} MODULE (candidate {label})\nStrengths: " + "; ".join(note.get("strengths", [])) + "\nWeaknesses: "
                    + "; ".join(note.get("weaknesses", [])) + f"\n```javascript\n{by[label].raw}\n```")

        text = prompts.MERGE_PICK + "\n\n" + block(base, "BASE") + "".join("\n\n" + block(label, "OTHER") for label in others)
        model, effort = self.plan.merge_call
        cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
        views: list[dict[str, Any]] = []
        call = Call(CRITIQUE_ID + 10 + offset, model, self.instructions,
                    [{"type": "input_text", "text": text}, self.target, *self.closeup, *views], effort, cap)
        merged = self.modules(pool, [call], ["dm" if not offset else f"dm{offset // 20 + 1}"])[0]
        if merged is not None and merged.uneven and not picked.uneven:  # uniform scale only
            return picked
        return merged if merged is not None else picked

    def fit_rest(self, base: list[Step], index: int, module: Candidate) -> list[Step]:
        rest = list(base[index:])
        chars = len(module.raw) * (1 + (MODULE_GROWTH - 1) * len(rest) / 9)
        size = chars / MODULE_CHARS_PER_TOKEN
        ratio = min(1.0, max(PATCH_FLOOR, sum(self.patch_ratios) / len(self.patch_ratios))) if self.patch_ratios else PATCH_PRIOR
        usable = CREDIT_LIMIT * (1 - SAFETY) - self.end_reserve
        text = int(len(self.instructions) + chars + len(prompts.REWRITE_PATCH) + 400)
        judge = worst_case(SOL, 4000, 9, SELECT_MAX_OUTPUT)

        def rewrite_cost(step: Step) -> float:
            model, effort = step.rewrites[0]
            reasoning, other = ROUND_COST.get((model, effort), ROUND_COST[(SOL, "xhigh")])
            weights = PRICING["models"][model]
            return (weights["input_credits_per_token"] * (other + size)
                    + weights["output_credits_per_token"] * (reasoning + size * (ratio if step.patch else 1.0)))


        def fits() -> bool:
            spent = self.budget.credits
            for step in rest:
                if not step.rewrites:
                    continue
                model, effort = step.rewrites[0]
                cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
                if spent + worst_case(model, text, 7, cap) > usable:
                    return False
                spent += rewrite_cost(step)
                if spent + 2 * judge > usable:
                    return False
                spent += 2 * SELECT_CALL_CREDITS
            return True

        xhigh = [i for i in range(len(rest) - 1, -1, -1) if rest[i].rewrites and rest[i].rewrites[0] == (SOL, "xhigh")]
        astra = [i for i in range(len(rest) - 1, -1, -1) if rest[i].rewrites and rest[i].rewrites[0][0] == ASTRA]
        lowered = len(xhigh) // 2
        moves = ([("high", i) for i in xhigh[:lowered]]
                 + [("xhigh", i) for i in astra] + [("high", i) for i in xhigh[lowered:]])
        for kind, i in moves:
            if fits():
                break
            rest[i] = replace(rest[i], rewrites=((SOL, kind),))
        return rest

    def draft_notes_each(self, pool: ThreadPoolExecutor, by: dict[str, Candidate], chosen: list[str], offset: int) -> dict[str, dict]:
        image = lambda b: {"type": "input_image", "image_url": data_url(b), "detail": "high"}
        calls = [Call(CRITIQUE_ID + offset + 1 + k, SOL, prompts.DRAFT_NOTES_SYSTEM,
                      [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup,
                       {"type": "input_text", "text": "CANDIDATE FRONT VIEWS:"}, image(by[label].front),
                       {"type": "input_text", "text": "CANDIDATE GRID:"}, image(by[label].sides),
                       {"type": "input_text", "text": prompts.DRAFT_NOTES_USER}],
                      "medium", SELECT_MAX_OUTPUT, text=prompts.DRAFT_NOTES_SCHEMA)
                 for k, label in enumerate(chosen)]
        found = {}
        for label, reply in zip(chosen, self.run_stage(pool, calls), strict=True):
            try:
                answer = json.loads(reply.output_text) if reply is not None else None
                found[label] = {"label": label, "strengths": [str(x) for x in answer["strengths"]][:6],
                                "weaknesses": [str(x) for x in answer["weaknesses"]][:6]}
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
        return found

    def read_text(self, pool: ThreadPoolExecutor) -> str:
        """TR: one Sol call that transcribes the photo's visible text; the block appended to the module-writing system text,
        or "" when there is none or the reply cannot be read."""
        call = Call(TEXT_READ_ID, SOL, prompts.TR_SYSTEM, [{"type": "input_text", "text": prompts.TR_USER}, self.target, *self.closeup],
                    "high", 16384, text=prompts.tr_schema())  # 2048 left no room after reasoning on text-heavy photos
        try:
            reply = self.run_stage(pool, [call])[0]
            items = json.loads(reply.output_text)["texts"] if reply is not None else []
        except (ValueError, TypeError, KeyError, AttributeError):
            return ""
        lines = []
        for item in items[:12]:
            text = str(item.get("text", "")).strip()
            if text and "illegible" not in text.lower() and not (text.startswith("[") and text.endswith("]")):
                lines.append(f'- "{text}" on {item.get("where", "")}; colour: {item.get("colour", "")}; style: '
                             f'{item.get("style", "")}; size: {item.get("size", "")}')
        if not lines:
            return ""
        return ("\n\nVISIBLE TEXT IN THE PHOTO (read once from the photo, letter by letter): reproduce every item below exactly as "
                "written here, letter by letter (keep unusual spellings), in the same place, colour, style and size, and keep all "
                "of it in every version of the module; never drop text that another version had.\n" + "\n".join(lines))

    def judge_pick(self, pool: ThreadPoolExecutor, parent: Candidate, child: Candidate, request_id: int, index: int = 0) -> bool:
        if len(parent.shots) < 2 or len(child.shots) < 2:
            return False
        if child.uneven and not parent.uneven:  # uniform scale only: the even parent stays
            self.gate_notes = ""
            return False
        image = lambda b: {"type": "input_image", "image_url": data_url(b), "detail": "high"}
        asks = ((parent.shots[0], child.shots[0], "slightly left of the front", True),
                (child.shots[1], parent.shots[1], "slightly right of the front", False))
        calls = [Call(request_id + offset, SOL, prompts.JUDGE_SYSTEM, [
            {"type": "input_text", "text": "Reference image (target object):"}, self.target,
            {"type": "input_text", "text": "3D model 1:"}, image(first),
            {"type": "input_text", "text": "3D model 2:"}, image(second),
            {"type": "input_text", "text": prompts.JUDGE_USER.format(angle_desc=angle)}], SELECT_EFFORT, JUDGE_MAX_OUTPUT,
            text=prompts.judge_schema()) for offset, (first, second, angle, _) in enumerate(asks)]
        gaps = []
        notes = []
        for (_, _, angle, parent_first), reply in zip(asks, self.run_stage(pool, calls), strict=True):
            try:
                answer = json.loads(reply.output_text) if reply is not None else None
                one, two = float(answer["penalty_1"]), float(answer["penalty_2"])
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            gaps.append(one - two if parent_first else two - one)
            notes.append((f"from {angle}", parent_first, str(answer.get("issues", ""))))
        kept = bool(gaps) and sum(gaps) / len(gaps) >= -JUDGE_DRAW
        self.gate_notes = gate_notes(notes, kept)
        return kept

    def fits(self, call: Call) -> bool:
        text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
        images = sum(1 for c in call.content if c["type"] == "input_image")
        return self.budget.allows([worst_case(call.model, text_chars, images, call.max_output_tokens)])

    def run_stage(self, pool: ThreadPoolExecutor, calls: list[Call]) -> list[Response | None]:
        admitted: list[Call | None] = []
        for call in calls:
            text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
            images = sum(1 for c in call.content if c["type"] == "input_image")
            call.estimate = worst_case(call.model, text_chars, images, call.max_output_tokens)
            admitted.append(call if self.budget.reserve(call.estimate) else None)
        futures = [pool.submit(self._dispatch, call) if call is not None else None for call in admitted]
        return [future.result() if future is not None else None for future in futures]

    def _dispatch(self, call: Call) -> Response | None:
        usage = None
        params: dict[str, Any] = {
            "model": call.model,
            "store": False,
            "service_tier": "default",
            "reasoning": {"effort": call.effort},
            "max_output_tokens": call.max_output_tokens,
            "instructions": call.instructions,
            "input": cast(ResponseInputParam, [{"role": "user", "content": call.content}]),
        }
        if call.text is not None:
            params["text"] = call.text
        try:
            result = self._create(call.request_id, params)
            usage = result.usage
            if usage is None:
                raise CallFailed(f"call {call.request_id}: reply without usage")
            return result if result.status == "completed" else None
        except APIConnectionError as error:
            raise
        except APIError as error:
            raise CallFailed(f"call {call.request_id}: {error}") from error
        except ValueError as error:
            if not any(sign in str(error).lower() for sign in CREDENTIAL_SIGNS):
                raise
            raise CallFailed(f"call {call.request_id}: {error}") from error
        finally:
            self.budget.settle(call.estimate, call.model, usage)

    def _create(self, request_id: int, params: dict[str, Any]) -> Response:
        waits = iter(OVERLOAD_WAITS)
        for attempt in range(JOIN_ATTEMPTS + len(OVERLOAD_WAITS)):
            try:
                return self.calls.create(request_id=request_id + self.salt, **params)
            except APITimeoutError:
                if attempt == JOIN_ATTEMPTS - 1:
                    raise
            except APIError as error:
                wait = next(waits, None)
                if wait is None or not any(sign in str(error) for sign in OVERLOADED):
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")

    def module_call(self, request_id: int, model: str, effort: str, extra: list[dict[str, Any]],
                    start: str = "plain") -> Call:
        target = prompts.TARGET_INVENTORY if start == "inventory" else prompts.TARGET_PROPORTIONS if start == "proportions" else prompts.TARGET
        content = [{"type": "input_text", "text": target}, self.target, *self.closeup, *extra]
        cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
        return Call(request_id, model, self.instructions, content, effort, cap)

    def rewrite_call(self, request_id: int, model: str, effort: str, parent: Candidate, focus: str = "", extra: str = "",
                     patch_edit: bool = False) -> Call:
        views = [{"type": "input_image", "image_url": data_url(view), "detail": "high"} for view in parent.views]
        call = self.module_call(request_id, model, effort, [
            {"type": "input_text", "text": "Current module, in order: 30 degrees left, 30 degrees right, from slightly below, from above; then the second-stage grid."},
            *views,
            {"type": "input_image", "image_url": data_url(parent.sides), "detail": "high"},
            {"type": "input_text", "text": prompts.rewrite_template("patch" if patch_edit else "full", self.plan.keep_dark).format(
                source=parent.raw, focus=prompts.REWRITE_FOCUS[focus] + extra)},
        ])
        if patch_edit:
            call.instructions = prompts.patch_instructions(self.instructions)
        return call

    def render_module(self, name: str, source: str, raw: str = "") -> Candidate | tuple[str, str]:
        try:
            front = self.renderer.render(source, FRONT_JUDGE)
            views = zoomed_views(front)
            grid = to_jpeg(self.renderer.render(source, GRID_VIEW))
        except RenderRejected as rejection:
            return source, rejection.detail_json
        candidate = Candidate(name, source, view_sheet(views), grid, raw, views)
        candidate.shots = judge_shots(front)[:2]
        candidate.uneven = uneven_scale(raw)
        return candidate

    def modules(self, pool: ThreadPoolExecutor, calls: list[Call], names: list[str],
                bases: dict[int, str] | None = None) -> list[Candidate | None]:
        results: list[Candidate | None] = [None] * len(calls)
        rejected: list[tuple[int, str, str]] = []
        for index, reply in enumerate(self.run_stage(pool, calls)):
            if reply is not None and bases is not None and index in bases:
                applied = patch.apply(bases[index], reply.output_text, extract_source)
                raw = applied.code if applied.code is not None and _EXPORT.search(applied.code) else None
                if raw is not None:
                    self.patch_ratios.append(1.0 if applied.full else _visible_tokens(reply) / max(1.0, len(bases[index]) / MODULE_CHARS_PER_TOKEN))
            else:
                raw = extract_source(reply.output_text) if reply is not None else None
            if raw is None:
                continue
            outcome = self.render_module(names[index], self.wrap(raw), raw)
            if isinstance(outcome, Candidate):
                outcome.notes = reply.output_text.split("```")[0].strip()[:2000]
                results[index] = outcome
            else:
                rejected.append((index, raw, outcome[1]))
        if self.renderer.over(CHAIN_ROOM):
            rejected = []
        repairs = [
            self.module_call(calls[index].request_id + REPAIR_OFFSET, SOL, calls[index].effort, [
                {"type": "input_text", "text": prompts.REPAIR.format(errors=errors, source=raw)}])
            for index, raw, errors in rejected
        ]
        for (index, _, _), reply in zip(rejected, self.run_stage(pool, repairs), strict=True):
            raw = extract_source(reply.output_text) if reply is not None else None
            if raw is not None:
                outcome = self.render_module(names[index], self.wrap(raw), raw)
                if isinstance(outcome, Candidate):
                    results[index] = outcome
        return results

    @staticmethod
    def parse_judgement(labels: dict[str, Candidate], replies: list[Response | None]) -> dict[str, Judgement]:
        scores: dict[str, list[float]] = {}
        fixes: dict[str, list[str]] = {}
        for reply in replies:
            if reply is None:
                continue
            try:
                entries = json.loads(reply.output_text)["candidates"]
            except (ValueError, KeyError, TypeError):
                continue
            seen = set()
            for entry in entries:
                label = entry.get("label")
                if label not in labels or label in seen:
                    continue
                seen.add(label)
                penalties = [entry[k] for k in ("front_left", "front_right", "front_below", "front_above")]
                score = sum(min(10, max(0, int(p))) for p in penalties) / 4
                if int(entry["garbage_sides"]) >= 3:
                    score += GARBAGE_PENALTY
                if "grid" in entry:
                    score += GRID_WEIGHT * min(10, max(0, int(entry["grid"])))
                scores.setdefault(label, []).append(score)
                fixes.setdefault(label, []).extend(str(f) for f in entry.get("fixes", [])[:6])
        return {
            labels[label].name: Judgement(sum(values) / len(values), list(dict.fromkeys(fixes[label]))[:8])
            for label, values in scores.items()
        }

    def scores(self, pool: ThreadPoolExecutor, candidates: list[Candidate], request_id: int,
               instructions: str = prompts.SELECT_CLOSE) -> dict[str, Judgement]:
        ordered = candidates.copy()
        random.Random(f"{self.seed}:{self.digest}:{request_id}").shuffle(ordered)
        labels = {chr(65 + i): candidate for i, candidate in enumerate(ordered)}
        maps = [labels, labels]
        sequences = [list(labels), list(reversed(labels))]
        calls = []
        for offset, (labels, order) in enumerate(zip(maps, sequences, strict=True)):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup]
            for label in order:
                content += [
                    {"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                    {"type": "input_image", "image_url": data_url(labels[label].front), "detail": "high"},
                    {"type": "input_text", "text": f"CANDIDATE {label} GRID:"},
                    {"type": "input_image", "image_url": data_url(labels[label].sides), "detail": "high"},
                ]
            calls.append(Call(request_id + offset, SOL, instructions, content, SELECT_EFFORT, SELECT_MAX_OUTPUT,
                              text=prompts.select_close_schema(list(labels))))
        replies = self.run_stage(pool, calls)
        judgements = self.parse_judgement(maps[0], replies)
        for candidate in candidates:
            if candidate.name in judgements:
                candidate.fixes = judgements[candidate.name].fixes
        return judgements

    def select(self, pool: ThreadPoolExecutor, candidates: list[Candidate], parents: set[str], keep: int,
               request_id: int) -> list[Candidate]:
        if len(candidates) <= keep:
            return candidates
        judgements = self.scores(pool, candidates, request_id)
        if not judgements and parents:
            return ([c for c in candidates if c.name in parents] + [c for c in candidates if c.name not in parents])[:keep]

        def rank(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            score = judgement.score if judgement else 99.0
            return (score + (PARENT_MARGIN if candidate.name in parents else 0.0), index)

        return [candidate for _, candidate in sorted(enumerate(candidates), key=rank)[:keep]]


    def run_stages(self, plan: StagePlan) -> str:
        self.plan = plan
        self.renderer.cap = plan.render_cap
        self.wrapper = chain_wrapper(plan.fill)
        self.end_reserve = 2 * worst_case(SOL, 4000, 11, SELECT_MAX_OUTPUT)
        closeup = object_closeup(self.image)
        if closeup is not None:
            self.closeup = [{"type": "input_image", "image_url": data_url(closeup), "detail": "high"}]
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="miner404") as pool:
            block = self.read_text(pool)  # TR: every module-writing call gets the photo's text, letter by letter
            if block:
                self.instructions = self.instructions + block
            routed = list(plan.steps)
            calls = [self.module_call(DRAFT_ID + i, draft.model, draft.effort, [], draft.start)
                     for i, draft in enumerate(plan.drafts)]
            names = [f"d{i}" for i in range(len(calls))]
            survivors = [c for c in self.modules(pool, calls, names) if c is not None]
            if not survivors:
                calls = [self.module_call(REDRAFT_ID + i, SOL, "high", [], draft.start)
                         for i, draft in enumerate(plan.drafts[:3])]
                survivors = [c for c in self.modules(pool, calls, [f"e{i}" for i in range(len(calls))]) if c is not None]
            if not survivors:
                return fallback_module(self.image)
            self.best_so_far = survivors[0].source
            drafts = survivors
            survivors = self.drafts_kept(pool, survivors, plan)
            survivors = [self.merged_pick(pool, drafts, survivors[0]), *survivors[1:]]
            for round_index in range(1, plan.merge_rounds):
                merged = survivors[0]
                survivors[0] = self.merged_pick(pool, [merged, *drafts], merged, 20 * round_index)
            self.best_so_far = survivors[0].source
            steps = list(routed)
            survivors = self.run_steps(pool, survivors, routed, steps, 0, len(steps))
            winner = self.select(pool, survivors, set(), 1, FINAL_SELECT_ID)[0]
            self.best_so_far = winner.source
            even = winner
            while even.uneven and even.parent is not None:  # uniform scale only: the last even module in the lineage
                even = even.parent
            if not even.uneven:
                winner = even
            elif winner.raw:  # every module of the lineage scales the whole object unevenly: drop that scale
                evened = even_scale(winner.raw)
                if evened != winner.raw:
                    outcome = self.render_module(winner.name + "e", self.wrap(evened), evened)
                    if isinstance(outcome, Candidate) and not outcome.uneven:
                        outcome.parent = winner.parent
                        winner = outcome
            self.best_so_far = winner.source
            return self.fitted_last(winner, plan)

    def run_steps(self, pool: ThreadPoolExecutor, survivors: list[Candidate], routed: list[Step], steps: list[Step],
                  start: int, stop: int) -> list[Candidate]:
        plan = self.plan
        for index in range(start, stop):
            if self.renderer.over(CHAIN_ROOM):
                break
            if index:
                steps[index:] = self.fit_rest(list(routed), index, survivors[0])
            step = steps[index]
            survivors = self.advance(pool, survivors, step, index)
            self.best_so_far = survivors[0].source
        return survivors

    def fitted_last(self, winner: Candidate, plan: StagePlan) -> str:
        if not winner.raw:
            return winner.source
        wrapper = chain_wrapper(plan.fill, True)
        source = self.wrap(winner.raw, wrapper)
        try:
            self.renderer.render(source, FRONT_JUDGE)
        except RenderRejected:
            return winner.source
        return source

    def drafts_kept(self, pool: ThreadPoolExecutor, drafts: list[Candidate], plan: StagePlan) -> list[Candidate]:
        ordered = self.select(pool, drafts, set(), len(drafts) - 1 if len(drafts) > plan.draft_keep else plan.draft_keep, SELECT_ID)
        ordered = [d for d in ordered if not d.uneven] + [d for d in ordered if d.uneven]  # uniform scale only: even drafts first
        if len(drafts) <= plan.draft_keep:
            return ordered
        kept, starts = [], set()
        for draft in ordered:
            start = plan.drafts[int(draft.name[1:])].start
            if start not in starts:
                starts.add(start)
                kept.append(draft)
        kept += [draft for draft in ordered if draft not in kept]
        return kept[:plan.draft_keep]

    def advance(self, pool: ThreadPoolExecutor, survivors: list[Candidate], step: Step, index: int) -> list[Candidate]:
        select_id = SELECT_ID + 10 * (index + 1)
        calls, names, owners = [], [], []
        bases: dict[int, str] = {}
        for position, parent in enumerate(survivors):
            focus = step.focus[position % len(step.focus)] if step.focus else ""
            for number, (model, effort) in enumerate(step.rewrites):
                request_id = REWRITE_BASE * (index + 1) + 10 * position + number
                extra = self.gate_notes if "gate" in step.info else ""
                call = self.rewrite_call(request_id, model, effort, parent, focus, extra, step.patch)
                if step.fallback and not self.fits(call):
                    call = self.rewrite_call(request_id, *step.fallback, parent, focus, extra, step.patch)
                if step.patch:
                    bases[len(calls)] = parent.raw
                calls.append(call)
                names.append(f"r{index}.{position}.{number}")
                owners.append(position)
        children = self.modules(pool, calls, names, bases if step.patch else None)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                child.parent = survivors[owner]
        kept = list(survivors)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                if self.judge_pick(pool, survivors[owner], child, select_id + 2 * owner, index):
                    kept[owner] = child
                self.history.append((index, child.notes, kept[owner] is child))
                self.changes.append((index, code_changes(survivors[owner].raw, child.raw), kept[owner] is child))
        return kept

    @staticmethod
    def best(candidates: list[Candidate], judgements: dict[str, Judgement]) -> Candidate:
        def key(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            return (judgement.score if judgement else 99.0, index)

        return min(enumerate(candidates), key=key)[1]


def run(image: bytes, seed: int, calls: ModelCalls, renderer: Renderer, plan: StagePlan) -> str | None:
    pipeline = Pipeline(image, seed, calls, renderer)
    try:
        return pipeline.run_stages(plan)
    except (APIConnectionError, CallFailed, WitnessError, ExchangePending, RenderUnavailable):
        raise
    except Exception:
        return pipeline.best_so_far or fallback_module(image)


ROUND_COST = {("gpt-6-astra", "high"): (494, 10313), ("gpt-6.1-sol", "xhigh"): (10712, 9125), ("gpt-6.1-sol", "high"): (748, 9575)}
SELECT_CALL_CREDITS = 15731
MODULE_GROWTH = 1.15


MODULE_CHARS_PER_TOKEN = 2.85
PATCH_FLOOR = 0.4
PATCH_PRIOR = 0.6


def uneven_scale(raw: str) -> bool:
    """Whether the model's module scales the whole object unevenly: a scale with unequal axes on the returned group or on
    the chain of groups that holds the whole object (each the only child of the one before). Children are read from the body
    of generate() only, so a helper function that names its own parameter like the returned group cannot hide the chain."""
    if not raw:
        return False
    top = "root"
    scope = raw
    start = raw.find("export default function")
    if start >= 0 and raw.find("{", start) >= 0:
        opened = raw.find("{", start)
        depth, end = 0, len(raw)
        for at in range(opened, len(raw)):  # the body of generate(): braces matched (strings with braces are rare here)
            if raw[at] == "{":
                depth += 1
            elif raw[at] == "}":
                depth -= 1
                if depth == 0:
                    end = at
                    break
        scope = raw[opened:end]
        returned = re.findall(r"\breturn\s+([A-Za-z_$][\w$]*)\s*;", scope)
        top = returned[-1] if returned else top
    kids: dict[str, list[str]] = {}
    for parent, args in re.findall(r"\b([A-Za-z_$][\w$]*)\.add\(([^()]*(?:\([^()]*\)[^()]*)*)\)", scope):
        for arg in args.split(","):
            arg = arg.strip()
            kids.setdefault(parent, []).append(arg if re.fullmatch(r"[A-Za-z_$][\w$]*", arg) else "<expr>")
    chain, name = [top], top
    while len(kids.get(name, [])) == 1 and kids[name][0] != "<expr>" and len(chain) < 6:
        name = kids[name][0]
        chain.append(name)
    for name in chain:
        n = re.escape(name)
        if re.search(rf"\b{n}\.scale\.[xyz]\s*=(?!=)", raw):
            return True  # one axis set on its own is uneven unless all three are set to the same number
        for args in re.findall(rf"\b{n}\.scale\.set\(([^()]*)\)", raw):
            values = [a.strip() for a in args.split(",")]
            if len(values) == 3 and len(set(values)) > 1:
                return True
    return False

def even_scale(raw: str) -> str:
    """The model's module with the unequal-axis scales removed from the whole-object chain (the groups uneven_scale checks),
    for the rare lineage in which every module scales the whole object unevenly. Each part keeps its own geometry; the
    wrapper's fit sizes the result uniformly."""
    if not raw or not uneven_scale(raw):
        return raw
    top = "root"
    scope = raw
    start = raw.find("export default function")
    if start >= 0 and raw.find("{", start) >= 0:
        opened = raw.find("{", start)
        depth, end = 0, len(raw)
        for at in range(opened, len(raw)):
            if raw[at] == "{":
                depth += 1
            elif raw[at] == "}":
                depth -= 1
                if depth == 0:
                    end = at
                    break
        scope = raw[opened:end]
        returned = re.findall(r"\breturn\s+([A-Za-z_$][\w$]*)\s*;", scope)
        top = returned[-1] if returned else top
    kids: dict[str, list[str]] = {}
    for parent, args in re.findall(r"\b([A-Za-z_$][\w$]*)\.add\(([^()]*(?:\([^()]*\)[^()]*)*)\)", scope):
        for arg in args.split(","):
            arg = arg.strip()
            kids.setdefault(parent, []).append(arg if re.fullmatch(r"[A-Za-z_$][\w$]*", arg) else "<expr>")
    chain, name = [top], top
    while len(kids.get(name, [])) == 1 and kids[name][0] != "<expr>" and len(chain) < 6:
        name = kids[name][0]
        chain.append(name)
    out = raw
    for name in chain:
        n = re.escape(name)
        out = re.sub(rf"\b{n}\.scale\.[xyz]\s*=(?!=)[^;\n]*;?", "", out)

        def drop(m: re.Match) -> str:
            values = [a.strip() for a in m.group(1).split(",")]
            return "" if len(values) == 3 and len(set(values)) > 1 else m.group(0)
        out = re.sub(rf"\b{n}\.scale\.set\(([^()]*)\)\s*;?", drop, out)
    return out


def even_scale(raw: str) -> str:
    """The model's module with the unequal-axis scales removed from the whole-object chain (the groups uneven_scale checks),
    for the rare lineage in which every module scales the whole object unevenly. Each part keeps its own geometry; the
    wrapper's fit sizes the result uniformly."""
    if not raw or not uneven_scale(raw):
        return raw
    top = "root"
    scope = raw
    start = raw.find("export default function")
    if start >= 0 and raw.find("{", start) >= 0:
        opened = raw.find("{", start)
        depth, end = 0, len(raw)
        for at in range(opened, len(raw)):
            if raw[at] == "{":
                depth += 1
            elif raw[at] == "}":
                depth -= 1
                if depth == 0:
                    end = at
                    break
        scope = raw[opened:end]
        returned = re.findall(r"\breturn\s+([A-Za-z_$][\w$]*)\s*;", scope)
        top = returned[-1] if returned else top
    kids: dict[str, list[str]] = {}
    for parent, args in re.findall(r"\b([A-Za-z_$][\w$]*)\.add\(([^()]*(?:\([^()]*\)[^()]*)*)\)", scope):
        for arg in args.split(","):
            arg = arg.strip()
            kids.setdefault(parent, []).append(arg if re.fullmatch(r"[A-Za-z_$][\w$]*", arg) else "<expr>")
    chain, name = [top], top
    while len(kids.get(name, [])) == 1 and kids[name][0] != "<expr>" and len(chain) < 6:
        name = kids[name][0]
        chain.append(name)
    out = raw
    for name in chain:
        n = re.escape(name)
        out = re.sub(rf"\b{n}\.scale\.[xyz]\s*=(?!=)[^;\n]*;?", "", out)

        def drop(m: re.Match) -> str:
            values = [a.strip() for a in m.group(1).split(",")]
            return "" if len(values) == 3 and len(set(values)) > 1 else m.group(0)
        out = re.sub(rf"\b{n}\.scale\.set\(([^()]*)\)\s*;?", drop, out)
    return out


def judge_shots(front: bytes) -> list[bytes]:
    image = Image.open(io.BytesIO(front)).convert("RGB")
    width, height = image.size
    shots = []
    for x, y in ((0, 0), (width // 2, 0), (0, height // 2), (width // 2, height // 2)):
        buffer = io.BytesIO()
        image.crop((x, y, x + width // 2, y + height // 2)).save(buffer, "PNG")
        shots.append(buffer.getvalue())
    return shots


def gate_notes(asks: list[tuple[str, bool, str]], kept_child: bool) -> str:
    other = "the version before it" if kept_child else "a rewrite the judge rejected"
    lines = []
    for where, parent_first, issues in asks:
        issues = issues.strip()
        if not issues:
            continue
        current = 1 if parent_first != kept_child else 2
        lines.append(f"- {where}: Model {current} is the current module, Model {3 - current} is {other}. {issues[:900]}")
    return prompts.GATE_NOTES.format(notes="\n".join(lines)) if lines else ""


def _visible_tokens(reply: object) -> int:
    usage = getattr(reply, "usage", None)
    if usage is None:
        return 0
    get = (lambda obj, key: obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None))
    out = get(usage, "output_tokens") or 0
    details = get(usage, "output_tokens_details")
    reasoning = (get(details, "reasoning_tokens") or 0) if details is not None else 0
    return max(0, int(out) - int(reasoning))


def code_changes(old: str, new: str, limit: int = 10) -> str:
    lines = [line for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
             if line[:1] in "+-" and not line.startswith(("+++", "---")) and line[1:].strip()]
    shown = [f"  {line[0]} {line[1:].strip()[:110]}" for line in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"  ... and {len(lines) - limit} more changed lines")
    return "\n".join(shown) if shown else "  (no code change)"
