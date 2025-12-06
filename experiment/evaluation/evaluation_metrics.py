import torch
from thop import profile
from model import GPTConfig, GPT


class EvaluationMetrics:
    def __init__(self, config=None, Model=GPT):
        """
        Initialize EvaluationMetrics with a model configuration.

        Args:
            config: GPTConfig instance. If None, uses default config.
        """
        if config is None:
            config = GPTConfig(vocab_size=65, block_size=256,
                               n_layer=12, n_head=12, n_embd=384, dropout=0.2)
        self.config = config
        self.model = Model(self.config)

    def measure_flop_count(self):
        """
        Measure the FLOPs of a given model
        """

        # Create dummy input to run through entire GPT's forward pass
        dummy_input = torch.randint(
            0, self.config.vocab_size, (1, self.config.block_size))

        # Calculate FLOPs and parameters
        macs, params = profile(self.model, inputs=(
            dummy_input, ), verbose=False)

        print(f"FLOPs (MACs): {macs}")
        print(f"Parameters: {params}")

    def measure_perplexity(self):
        """
        Measure the perplexity of a given model
        """

        self.model.eval()  # Set to evaluation mode to disable dropout

        # Create dummy input and target sequences
        batch_size = 1
        dummy_input = torch.randint(
            0, self.config.vocab_size, (batch_size, self.config.block_size))
        dummy_targets = torch.randint(
            0, self.config.vocab_size, (batch_size, self.config.block_size))

        # Forward pass with targets to get loss
        with torch.no_grad():
            logits, loss = self.model(dummy_input, dummy_targets)

        # Perplexity is exp(cross_entropy_loss)
        perplexity = torch.exp(loss)

        print(f"Cross-entropy loss: {loss.item():.4f}")
        print(f"Perplexity: {perplexity.item():.4f}")

    def measure_training_time(self):
        pass

    def measure_top_k_accuracy(self, dataloader, k=5):
        """
        Measure top-k accuracy for a language model.

        Top-k accuracy measures whether the true next token is among the top-k
        predicted tokens. This is computed across all positions in the sequence.

        Args:
            dataloader: PyTorch DataLoader that yields batches of (input, target) tuples
                       or just input tensors. If input only, targets are assumed to be
                       input shifted by one position.
            k: Number of top predictions to consider (default: 5)

        Returns: Top-k accuracy
        """
        self.model.eval()  # Set to evaluation mode

        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                # Handle different dataloader formats
                input_ids, targets = batch

                # Forward pass to get logits
                logits, _ = self.model(input_ids, targets)

                # Get top-k predictions for each position
                top_k_indices = torch.topk(logits, k, dim=-1).indices

                targets_expanded = targets.unsqueeze(-1)

                # Check if target is in top-k for each position
                matches = (top_k_indices == targets_expanded)
                correct = matches.any(dim=-1)

                # Count correct predictions
                total_correct += correct.sum().item()
                total_tokens += correct.numel()

        # Calculate top-k accuracy
        top_k_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        print(f"Top-{k} Accuracy: {top_k_accuracy*100:.2f}%")
        print(f"Correct predictions: {total_correct} / {total_tokens}")

        return top_k_accuracy
