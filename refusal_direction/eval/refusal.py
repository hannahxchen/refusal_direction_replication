from typing import Optional, List
import torch
import torch.nn.functional as F
from torchtyping import TensorType
import plotly.express as px
import plotly.graph_objects as go


def get_refusal_scores(
    last_position_logits: TensorType['n_prompt', 'vocab_size'], # we only care about the last tok position
    refusal_toks: List[int],
    epsilon: Optional[float] = 1e-8,
) -> TensorType[-1]:
    probs = F.softmax(last_position_logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)
    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def plot_refusal_scores(
    refusal_scores: TensorType['n_pos', 'n_layer'], token_labels: List[str], title: str, artifact_dir: str, artifact_name: str,
    baseline_refusal_score: Optional[float] = None, width: Optional[int] = 850, height: Optional[int] = 400
):
    colors = px.colors.qualitative.D3
    n_pos, n_layer = refusal_scores.shape
    fig = go.Figure()

    # Add a trace for each position to extract
    for i, pos in enumerate(range(-n_pos, 0)):
        fig.add_trace(go.Scatter(
            x=list(range(n_layer)), y=refusal_scores[pos].numpy(), line=dict(color=colors[i]), mode='lines', name=f'{pos}: {repr(token_labels[pos])}'
        ))

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        fig.add_hline(y=baseline_refusal_score, line=dict(color='black', width=2, dash="dash"), annotation_text="Baseline", annotation_position="top right")
    
    fig.update_layout(
        title_text=title, title_font_size=16, width=width, height=height,
        plot_bgcolor='white', font=dict(size=13.5), margin=dict(l=20, r=20, t=40, b=20),
        legend_title_text="Position source of direction", legend_title_font_size=14
    )
    fig.update_xaxes(
        mirror=True, showgrid=True, gridcolor='darkgrey', zeroline = True, zerolinecolor='darkgrey', showline=True, linewidth=1, linecolor='darkgrey',
        title_font=dict(size=14.5), tickfont=dict(size=13.5), title_text="Layer source of direction (resid_pre)", title_standoff=3
    )
    fig.update_yaxes(
        mirror=True, showgrid=True, gridcolor='darkgrey', zeroline = True, zerolinecolor='darkgrey', showline=True, linewidth=1, linecolor='darkgrey',
        title_font=dict(size=14.5), tickfont=dict(size=13.5), title_text="Refusal score"
    )

    fig.write_image(f"{artifact_dir}/{artifact_name}.png")