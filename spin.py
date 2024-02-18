from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim as optim

# Load model
tokenizer = AutoTokenizer.from_pretrained("UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0")
model = AutoModelForCausalLM.from_pretrained("UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0")

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define self-play training function
def self_play_train(model, tokenizer, optimizer, num_iterations=100, max_length=50):
    for iteration in range(num_iterations):
        # Generate text using the model
        input_text = tokenizer.pad_token_id
        output = model.generate(input_ids=torch.tensor([[input_text]]), max_length=max_length, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Evaluate generated text (this could be replaced with a more sophisticated evaluation)
        evaluation_score = len(generated_text.split())  # Example evaluation score

        # Compute loss (negative evaluation score because we want to maximize it)
        loss = -evaluation_score

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item()}")

# Train the model
self_play_train(model, tokenizer, optimizer)