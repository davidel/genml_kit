"""Declarative model of the external services the README CLI suite touches.

A marked README fence that talks to a third-party service (HuggingFace,
AWS S3, Cloudflare R2, Google Cloud Storage, GitHub, Weights & Biases)
names that service with the ``service:`` marker key.  The suite runs the
fence when the service is *available* -- and a service is available when
either

* the credentials it needs are present in the environment, **or**
* an endpoint override is configured (a proxy / emulator), because every
  underlying client library honours its own override variable natively and
  ``genml_kit`` never has to be told about it.

That second path lets a sandbox that proxies the internet -- or a CI job
that stands up an emulator -- exercise these fences without holding any
real secret.  The mapping below is the single source of truth for both the
gate (:func:`require`) and the pytest markers the harness emits.
"""

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Service:
  """One external dependency and the env vars that make it usable.

    Attributes:
        name: Marker key used in the README fence, e.g. ``"hf"``.
        credentials: Alternative credential groups.  Each group is a tuple
            of env vars that must *all* be present (e.g. an access/secret
            pair); any single group being satisfied makes the service
            usable.  A single-variable group models a lone token, and two
            groups model interchangeable tokens (e.g. ``HF_TOKEN`` and
            ``HUGGING_FACE_HUB_TOKEN``).
        endpoints: Env vars that point the client at a proxy or emulator.
            Any single one being set satisfies the service; this is a list
            of alternatives, not a conjunction.
  """

  name: str
  credentials: tuple[tuple[str, ...], ...] = ()
  endpoints: tuple[str, ...] = ()


SERVICES = {
    "hf":
        Service("hf", (("HF_TOKEN",), ("HUGGING_FACE_HUB_TOKEN",)), ("HF_ENDPOINT",)),
    "s3":
        Service("s3", (("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),),
                ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL")),
    "r2":
        Service("r2", (("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"),),
                ("R2_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3")),
    "gcs":
        Service("gcs", (("GOOGLE_APPLICATION_CREDENTIALS",),),
                ("STORAGE_EMULATOR_HOST",)),
    "github":
        Service("github", (("GITHUB_TOKEN",), ("GH_TOKEN",)), ("GITHUB_API_URL",)),
    "wandb":
        Service("wandb", (("WANDB_API_KEY",),), ("WANDB_BASE_URL",)),
}


def _env_present(env, name):
  return bool((env.get(name) or "").strip())


def has_credentials(name, env):
  """Return ``True`` when any credential group for *name* is fully set."""
  service = SERVICES[name]
  return any(
      all(_env_present(env, var) for var in group) for group in service.credentials)


def endpoint_override(name, env):
  """Return the first endpoint-override var set for *name*, or ``None``."""
  for var in SERVICES[name].endpoints:
    if _env_present(env, var):
      return var
  return None


def available(name, env):
  """Return ``True`` when *name* is usable via credentials or an override."""
  return has_credentials(name, env) or endpoint_override(name, env) is not None


def skip_reason(name):
  """Human-readable reason for skipping an unavailable *name*."""
  service = SERVICES[name]
  groups = [" + ".join(group) for group in service.credentials]
  parts = groups + list(service.endpoints)
  if not parts:
    return f"requires the {name!r} service"
  return "requires " + " or ".join(parts)


def require(name, env):
  """Skip the current test unless *name* is available in *env*."""
  if not available(name, env):
    pytest.skip(skip_reason(name))
