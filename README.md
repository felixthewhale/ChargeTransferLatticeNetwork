## ChargeTransferLatticeNetwork

It uses a grid where "signals" or "energy" flows locally between points based on learned rules (the weights).

## Architecture Overview: The Core Concepts

At the core of this experimental architecture is a **ChargeTransferLatticeNetwork (CTN)**. Envision it as a 3D grid where 'charge' dynamically flows between cells, serving as the model's primary computational engine. We control the movement of this charge (the CTN's weights), but the internal dynamics evolve naturally based on these weights and the input.
This architecture in this example is applied to the token generation.

*   **The 'Thought' State:** As the model generates each token, the CTN processes internally, evolving its dynamic state. This instantaneous internal configuration of the CTN is what we refer to as the model's 'thought' at any given moment.
*   **Self-Modeling and Internal Coherence:** A key aspiration is for the model to develop an internal representation of its own processing. This is achieved by requiring the model to predict its *own* subsequent internal state (its next 'thought'). This predictive auxiliary task encourages the CTN to develop more predictable and coherent internal dynamics, fostering consistent and potentially more sophisticated behavioral patterns.
*   **Taking in Input:** For each input word, its embedding is sent to a special 'input' face of the CTN (currently the `x=0` face). Here, it adds to the existing charge pattern.
*   **Remembering Inner State:** A key feature is how the CTN's state keeps going. The full 3D CTN state from finishing one word's processing becomes the *starting state* for the next word. This lets 'thought' carry over and develop through the whole piece of text, building a steady inner context.
*   **Inner Change (`ctn_iterations_per_token`):** After getting input, the CTN changes its inner charge pattern for a set number of internal steps (`ctn_iterations_per_token`). During these internal steps, the autoencoder and latent state predictor are always watching and shaping the 'thought' state.
*   **Guessing the Next Word:** After the CTN finishes its inner work for a given word, its final 3D state (the refined 'thought' for that step) is turned into its small number-form (`z_K`). This `z_K` then goes into a final network (`token_predictor_head_from_z`) which guesses the next language word.

## How it Works: Architectural Components

1.  **Thought State Autoencoder:** An autoencoder is employed to compress the high-dimensional 3D state of the CTN into a compact, lower-dimensional latent representation. This compression forces the model to identify features of its evolving internal state.
2.  **Latent State Prediction Module:** This component is trained to predict the temporal evolution of the compressed latent thought state. Its purpose is to model the internal dynamics, providing a self-reference objective.

## Todo (To Test)

* Learnable transfer?
* 4D History?
* Multiple channels of small CTNs 

## Training

*   **Combined Training Objective:** The model is optimized simultaneously for two distinct, yet complementary, objectives:
    *   **Primary Task:** Predicting the next token in a sequence.
    *   **Auxiliary Task:** Predicting its own subsequent 'thought' state (the latent CTN state).

*   **Dataset:** To be tested
