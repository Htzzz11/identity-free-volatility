# Same Company, Different Quarter

Isolating call-specific volatility signal in earnings call transcripts.
COMS 4705 (Natural Language Processing), Columbia, Fall 2026.

Yanshengtian Dong · Songhan Wu · Tianzhi Hou

## Question

Yu, Liu and He (ACL Findings 2025) show that transcript models for
post-earnings volatility mostly learn company identity: a per-ticker historical
mean (STPEV) beats them, and random per-ticker vectors match real embeddings.
We ask how much volatility signal a call carries **after** identity is removed
from its representation, and whether identity-removal training recovers signal
that standard training hides.

- **H1** A model trained on the residual target `v - STPEV_c` beats STPEV.
- **H2** Domain-adversarial training (ticker as domain) lowers post-hoc ticker
  probe accuracy without raising residual error.
- **H3** Representations of the change between consecutive calls carry more
  call-specific signal than single-call representations.

Targets: log volatility over the next 3, 7, 15, 30 trading days.
Metrics: MSE per horizon (DEC rolling-by-quarter protocol), Spearman on
residuals, post-hoc ticker probe accuracy. Three seeds per configuration.
