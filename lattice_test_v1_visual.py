import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

class ChargeTransferLatticeNetwork(nn.Module):
    def __init__(self, width=16, height=16, depth=32, num_iterations=50):
        """
        Initializes a Lattice Network based on charge transfer dynamics.

        Args:
            width, height, depth, num_iterations: As before.
        """
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.num_iterations = num_iterations
        self.epsilon = 1e-9 # For stable division

        # --- Trainable Weights ---
        # Represent propensity/rate of transfer (0 to 1 range is useful)
        # Initialize weights near zero initially for slow, controlled transfer
        self.weights = nn.Parameter(torch.randn(6, width, height, depth) * 0.1 - 2.0) # Centered around sigmoid(x)=-2 ~= 0.1
        # Bias term doesn't make much sense in this model, omitting it.

        self._intermediate_states = []
        self._record_intermediates = False

    def forward(self, input_signal, record_intermediates=False):
        """ Runs the lattice dynamics using charge transfer. """
        self._record_intermediates = record_intermediates
        if record_intermediates:
            self._intermediate_states = []

        # --- Input Shape Handling & Device ---
        if input_signal.dim() == 2: input_signal = input_signal.unsqueeze(0)
        # ... (rest of shape checks) ...
        batch_size = input_signal.shape[0]
        device = input_signal.device
        W, H, D = self.width, self.height, self.depth

        # --- Initialize State (Assume non-negative charge) ---
        state = torch.zeros(batch_size, W, H, D, device=device)
        state[:, 0, :, :] = torch.relu(input_signal) # Ensure initial input is non-negative
        # state[:, 0, :, :] = input_signal # Allow negative if desired

        if self._record_intermediates:
            self._intermediate_states.append(state.detach().cpu().numpy())

        # Pre-calculate sigmoid of weights once if not learning
        # If learning, do this inside loop or ensure weights don't change mid-inference
        transfer_rates = torch.sigmoid(self.weights) # Values between 0 and 1

        # --- Run Iterations ---
        for i in range(self.num_iterations):
            # Ensure state is non-negative for this model if needed
            current_state_nonneg = torch.relu(state)
            # current_state_nonneg = state # If allowing negative state

            # --- 1. Calculate Potential Outflows for each direction ---
            # outflows[dir] contains potential flow FROM (x,y,z) IN direction 'dir'
            outflows = torch.zeros(6, batch_size, W, H, D, device=device)
            outflows[0] = current_state_nonneg * transfer_rates[0][None, ...] # Flow to +x
            outflows[1] = current_state_nonneg * transfer_rates[1][None, ...] # Flow to -x
            outflows[2] = current_state_nonneg * transfer_rates[2][None, ...] # Flow to +y
            outflows[3] = current_state_nonneg * transfer_rates[3][None, ...] # Flow to -y
            outflows[4] = current_state_nonneg * transfer_rates[4][None, ...] # Flow to +z
            outflows[5] = current_state_nonneg * transfer_rates[5][None, ...] # Flow to -z

            # --- 2. Limit Outflows to Conserve Charge ---
            total_potential_outflow = torch.sum(outflows, dim=0) # Sum over 6 directions
            # Calculate scaling factor (ensure we don't send more than we have)
            scale = torch.min(torch.ones_like(current_state_nonneg),
                              current_state_nonneg / (total_potential_outflow + self.epsilon))
            # Apply scaling to get actual outflows
            actual_outflows = outflows * scale[None, ...] # Expand scale dims to match outflows

            # --- 3. Calculate Total Actual Outflow from each node ---
            total_actual_outflow = torch.sum(actual_outflows, dim=0)

            # --- 4. Calculate Total Inflow to each node ---
            # Inflow arrives from the opposite direction of a neighbor's outflow
            total_inflow = torch.zeros_like(state)
            # Inflow from left (x-1) = Outflow[+x] from (x-1)
            if W > 1: total_inflow[:, 1:, :, :] += actual_outflows[0, :, :-1, :, :]
            # Inflow from right (x+1) = Outflow[-x] from (x+1)
            if W > 1: total_inflow[:, :-1, :, :] += actual_outflows[1, :, 1:, :, :]
            # Inflow from up (y-1) = Outflow[+y] from (y-1)
            if H > 1: total_inflow[:, :, 1:, :] += actual_outflows[2, :, :, :-1, :]
            # Inflow from down (y+1) = Outflow[-y] from (y+1)
            if H > 1: total_inflow[:, :, :-1, :] += actual_outflows[3, :, :, 1:, :]
            # Inflow from front (z-1) = Outflow[+z] from (z-1)
            if D > 1: total_inflow[:, :, :, 1:] += actual_outflows[4, :, :, :, :-1]
            # Inflow from back (z+1) = Outflow[-z] from (z+1)
            if D > 1: total_inflow[:, :, :, :-1] += actual_outflows[5, :, :, :, 1:]

            # --- 5. Update State ---
            change = total_inflow - total_actual_outflow
            state = state + change

            # Optional: Add passive decay?
            # state = state * 0.99 + change

            if self._record_intermediates:
                self._intermediate_states.append(state.detach().cpu().numpy())

        # --- Extract Output ---
        output_signal = state[:, -1, :, :]
        return output_signal

    def get_intermediate_states(self):
        return self._intermediate_states

def visualize_lattice_slice(states_list, slice_y_index, title):
    """
    Visualizes the evolution of a 3D lattice state over time by showing
    an animated 2D slice along the Y (Height) axis.

    Args:
        states_list (list): A list of NumPy arrays, where each array represents
                            the state at a specific time step. Expected shape:
                            (batch_size, width, height, depth).
        slice_y_index (int): The index along the height (Y) axis to visualize.
        title (str): The title for the plot and animation.
    """
    if not states_list:
        print("Warning: No states provided for visualization.")
        return

    # Assume batch_index = 0 for simplicity, as it wasn't specified
    batch_index = 0
    first_state = states_list[0]

    # Basic validation
    if first_state.ndim != 4:
        raise ValueError(f"Expected states with 4 dimensions (batch, W, H, D), but got {first_state.ndim}")

    B, W, H, D = first_state.shape

    if not (0 <= batch_index < B):
         raise ValueError(f"Internal error: batch_index {batch_index} out of bounds for size {B}")
    if not (0 <= slice_y_index < H):
        raise ValueError(f"slice_y_index {slice_y_index} is out of bounds for height {H}")

    print(f"Visualizing Y-slice at index={slice_y_index}, batch element={batch_index}")

    # Extract all relevant slices (these will be the X-Z planes at the specified Y index)
    all_slices = []
    for i, state in enumerate(states_list):
         if state.ndim == 4 and state.shape[0] > batch_index and state.shape[2] > slice_y_index:
             slice_data = state[batch_index, :, slice_y_index, :] # Shape (W, D)
             all_slices.append(slice_data)
         else:
             print(f"Warning: Skipping state at index {i} due to unexpected shape/index: {state.shape}")


    if not all_slices:
        print("Warning: No valid slices could be extracted.")
        return

    # Determine global min/max across all time steps for consistent color scaling
    try:
        global_min = np.min([s.min() for s in all_slices if s.size > 0])
        global_max = np.max([s.max() for s in all_slices if s.size > 0])
        # Add a small buffer if min and max are the same (e.g., all zeros)
        if global_min == global_max:
            global_min -= 0.1
            global_max += 0.1
        if global_min == global_max: # Still same (e.g., all zeros)
             global_max += 1.0 # Avoid zero range for colorbar
    except ValueError: # Handle case where all_slices might be empty after filtering
        print("Warning: Could not determine color limits.")
        global_min, global_max = 0, 1 # Default fallback

    # Setup the plot
    fig, ax = plt.subplots()
    initial_slice = all_slices[0] # Shape (W, D)

    # Displaying the X-Z plane.
    # imshow plots row vs column index.
    # We have slice_data[W, D]. imshow(slice_data) maps W->rows(vertical), D->cols(horizontal).
    # X (Width) on horizontal and Z (Depth) on vertical, transpose.
    # imshow(slice_data.T) maps D->rows(vertical), W->cols(horizontal). Correct!
    cmap = 'viridis' # like 'magma', 'plasma', 'inferno'
    im = ax.imshow(initial_slice.T, cmap=cmap, vmin=global_min, vmax=global_max,
                   origin='lower', interpolation='nearest', aspect='auto',
                   extent=[0, W, 0, D]) # extent is [left, right, bottom, top] -> [X_min, X_max, Z_min, Z_max]

    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Depth (Z)")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('State Value / Charge Density')

    # Animation update function
    def update(frame):
        slice_data = all_slices[frame]
        im.set_data(slice_data.T) # Transpose for correct orientation in imshow
        # Update title with time step - append to the provided title
        ax.set_title(f"{title} (Time Step {frame}/{len(all_slices)-1})")
        # Note: We don't return ax.title, only the artists that changed (im)
        return [im]

    # Create animation
    interval = 50 # ms
    ani = animation.FuncAnimation(fig, update, frames=len(all_slices),
                                  interval=interval, blit=True, repeat_delay=1000)

    # Display the animation
    plt.show()

    # not strictly necessary for just displaying it.
    # return ani

# --- Example Usage ---
if __name__ == "__main__":
    W, H, D = 16, 16, 32
    N_ITER = 50 # Maybe shorter needed now? Or longer to see bounces.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChargeTransferLatticeNetwork(width=W, height=H, depth=D,
                                         num_iterations=N_ITER).to(device)
    print(f"ChargeTransferModel created: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Dummy Input (Gaussian pulse might be good) ---
    input_h = torch.linspace(-1, 1, H)
    input_d = torch.linspace(-1, 1, D)
    grid_h, grid_d = torch.meshgrid(input_h, input_d, indexing='ij')
    center_h, center_d = H // 2, D // 2
    sigma = 3.0
    # Gaussian pulse, ensuring it's positive
    dummy_input = torch.exp(-((grid_h - input_h[center_h])**2 + (grid_d - input_d[center_d])**2) / (2 * sigma**2)) * 1.0 # Peak at 1
    dummy_input = dummy_input.to(device)

    print("Running inference with charge transfer...")
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        output = model(dummy_input, record_intermediates=True)
    end_time = time.time()
    print(f"Inference finished in {end_time - start_time:.2f} seconds.")
    print("Output shape:", output.shape)

    intermediate_states = model.get_intermediate_states()
    if intermediate_states:
        print(f"Number of recorded states: {len(intermediate_states)}") # N_ITER + 1
        visualize_lattice_slice(intermediate_states, slice_y_index=H // 2,
                                title=f"Charge Transfer Slice (Y={H//2})")
