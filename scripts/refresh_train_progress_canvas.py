#!/usr/bin/env python3
"""Refresh the ACP training progress canvas from live logs / process state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_progress_server import job_status  # noqa: E402

CANVAS = Path.home() / ".cursor/projects/home-zj-ACP-fx/canvases/acp-train-progress.canvas.tsx"


def downsample(losses: list[dict], n: int = 40) -> list[dict]:
    if len(losses) <= n:
        return losses
    step = max(1, len(losses) // n)
    out = losses[::step]
    if out[-1] != losses[-1]:
        out.append(losses[-1])
    return out


def main() -> None:
    d = job_status()
    flip, vase = d["flip"], d["vase"]
    losses = downsample(flip["losses"])
    cats = [str(x["epoch"]) for x in losses]
    vals = [round(float(x["loss"]), 5) for x in losses]

    flip_loss = "null" if flip["loss"] is None else repr(float(flip["loss"]))
    vase_epoch = "null" if vase["epoch"] is None else repr(int(vase["epoch"]))
    vase_loss = "null" if vase["loss"] is None else repr(float(vase["loss"]))
    eta = "null" if flip["eta_hours"] is None else repr(float(flip["eta_hours"]))

    canvas = f"""import {{
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Text,
  UsageBar,
}} from "cursor/canvas";

const UPDATED = {json.dumps(d["updated_at"])};
const FLIP = {{
  status: {json.dumps(flip["status"])},
  epoch: {json.dumps(flip["epoch"])},
  loss: {flip_loss},
  pct: {float(flip["pct"])},
  etaHours: {eta},
}};
const VASE = {{
  status: {json.dumps(vase["status"])},
  epoch: {vase_epoch},
  loss: {vase_loss},
  pct: {float(vase["pct"])},
}};
const LOSS_EPOCHS = {json.dumps(cats)};
const LOSS_VALUES = {json.dumps(vals)};

function statusTone(s: string): "success" | "warning" | "neutral" | "info" {{
  if (s === "running") return "success";
  if (s === "queued") return "warning";
  if (s === "done") return "info";
  return "neutral";
}}

export default function AcpTrainProgress() {{
  const flipLoss =
    FLIP.loss == null ? "—" : (FLIP.loss as number).toFixed(4);
  const eta =
    FLIP.etaHours == null ? "—" : `${{FLIP.etaHours as number}} h`;
  const vaseBody =
    VASE.status === "queued"
      ? "Queued: starts automatically when flip_up finishes."
      : VASE.epoch == null
        ? "Waiting to start."
        : `Epoch ${{VASE.epoch}} · loss ${{VASE.loss}}`;

  return (
    <Stack gap={{20}} style={{{{ padding: 16 }}}}>
      <Stack gap={{4}}>
        <H1>ACP Training Progress</H1>
        <Text tone="secondary">
          Live browser: http://127.0.0.1:8765 · snapshot {{UPDATED}}
        </Text>
      </Stack>

      <UsageBar
        total={{100}}
        topLeftLabel={{`flip_up ${{FLIP.pct}}%`}}
        topRightLabel="300 epochs"
        segments={{[
          {{ id: "done", value: FLIP.pct }},
        ]}}
      />

      <Grid columns={{2}} gap={{16}}>
        <Card>
          <CardHeader
            title="flip_up_230"
            trailing={{
              <Pill tone={{statusTone(FLIP.status)}}>
                {{FLIP.status.toUpperCase()}}
              </Pill>
            }}
          />
          <CardBody>
            <Row gap={{24}}>
              <Stat
                label="Epoch"
                value={{
                  FLIP.epoch == null ? "—" : `${{FLIP.epoch}} / 299`
                }}
              />
              <Stat label="Train loss" value={{flipLoss}} />
              <Stat label="ETA" value={{eta}} />
            </Row>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="vase_wiping_200"
            trailing={{
              <Pill tone={{statusTone(VASE.status)}}>
                {{VASE.status.toUpperCase()}}
              </Pill>
            }}
          />
          <CardBody>
            <Stack gap={{8}}>
              <Text>{{vaseBody}}</Text>
              <Text tone="secondary">
                Dataset ready at ~/data/real_processed/vase_wiping_v6.3
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={{8}}>
        <H2>flip_up train loss</H2>
        <Text tone="secondary">
          Source: training_outputs logs.json.txt · downsampled · y = train_loss
        </Text>
        <LineChart
          categories={{LOSS_EPOCHS}}
          series={{[{{ name: "train_loss", data: LOSS_VALUES, tone: "info" }}]}}
          height={{220}}
          fill
        />
      </Stack>
      <Spacer />
    </Stack>
  );
}}
"""
    CANVAS.write_text(canvas)
    print(
        f"canvas updated {d['updated_at']} "
        f"flip={flip['status']} epoch={flip['epoch']} pct={flip['pct']}% "
        f"vase={vase['status']}"
    )


if __name__ == "__main__":
    main()
