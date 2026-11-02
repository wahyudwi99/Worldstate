# Deferred controller phase

The MVP ends after VAE representation learning, MDN-RNN dynamics, and held-out
imagined-rollout evaluation. Implement a controller only after an explicit request.

A future controller would concatenate the current latent z_t with the LSTM hidden
state summarizing prior transitions, then apply a linear layer. Map steering with
tanh and gas/brake with sigmoid. After selecting action a_t, update the memory
with (z_t, a_t). Reset both hidden and cell state at every environment reset.

Optimize controller weights with CMA-ES using rewards from the real simulator.
This is evolutionary policy optimization, not supervised action prediction.
Do not optimize rewards inside this MVP's dream model: it has neither a reward
head nor a termination head, and model exploitation would require separate work.

Budget simulator evaluations explicitly: population size times generations times
episodes per candidate. Even a small search can require thousands of episodes.
First benchmark one episode, propose an evaluation budget, and obtain an explicit
request before starting controller training. Keep evolution seeds separate from
final held-out policy evaluation seeds. No paper-level score is promised.
