def sample_real_indices_sliced(self, target_labels):
    """
    Sample indices of real images with labels in the sliced vicinity of target_labels
    using multiple projection vectors.
    
    Parameters:
    - target_labels: Target labels to find vicinity samples for [B, D]
    
    Returns:
    - batch_real_indx: Indices of selected real samples [B]
    """
    batch_size = len(target_labels)
    batch_real_indx = np.zeros(batch_size, dtype=int)
    
    # Get label dimension
    dim = self.label_dim
    device = target_labels.device
    
    # Generate random vectors for projection (use multiple vectors)
    v_all = generate_random_vectors(
        self.vector_type, dim, self.num_projections, device
    )
    
    # Convert training labels to tensor once
    train_labels_tensor = torch.from_numpy(self.train_labels).float().to(device)
    
    # Process each target label
    for j in range(batch_size):
        target_label = target_labels[j:j+1]  # Keep dimension [1, D]
        
        # Track matched indices across all projections
        all_matched_indices = []
        
        # Try each projection vector
        for v_idx in range(self.num_projections):
            v = v_all[v_idx:v_idx+1]  # [1, D]
            
            # Normalize projection vector
            v_norm = F.normalize(v, dim=1)
            
            # Project all training labels and target label
            proj_train_labels = torch.matmul(train_labels_tensor, v_norm.t()).squeeze(-1)  # [N]
            proj_target_label = torch.matmul(target_label, v_norm.t()).squeeze(-1)  # [1]
            
            # Calculate projection differences
            proj_diff = torch.abs(proj_train_labels - proj_target_label)
            
            # Adjust kappa based on vector norm for numerical stability
            effective_kappa = self.kappa * torch.norm(v)
            
            # Find indices within vicinity
            indx_real_in_vicinity = torch.where(proj_diff <= effective_kappa)[0]
            
            # Store found indices
            if len(indx_real_in_vicinity) > 0:
                all_matched_indices.append(indx_real_in_vicinity)
        
        # Combine all matched indices
        if len(all_matched_indices) > 0:
            # Concatenate all indices
            combined_indices = torch.cat(all_matched_indices)
            
            # Count frequency of each index (samples that match multiple projections are preferred)
            unique_indices, counts = torch.unique(combined_indices, return_counts=True)
            
            # Sort by frequency (descending)
            sorted_indices = torch.argsort(counts, descending=True)
            best_indices = unique_indices[sorted_indices]
            
            # Select top-k indices based on frequency
            top_k = min(10, len(best_indices))
            candidate_indices = best_indices[:top_k]
            
            # Randomly select one from the top candidates
            chosen_idx = torch.randint(0, len(candidate_indices), (1,), device=device)[0]
            batch_real_indx[j] = candidate_indices[chosen_idx].cpu().numpy()
        else:
            # Fallback: If no matches found across all projections, use nearest neighbor
            if train_labels_tensor.dim() > 1 and train_labels_tensor.shape[1] > 1:
                # Multi-dimensional case
                diff = train_labels_tensor - target_label
                dist = torch.sqrt((diff ** 2).sum(dim=1))
            else:
                # 1D case
                dist = torch.abs(train_labels_tensor - target_label)
            
            # Get index of closest training sample
            closest_idx = torch.argmin(dist)
            batch_real_indx[j] = closest_idx.cpu().numpy()
    
    return batch_real_indx
