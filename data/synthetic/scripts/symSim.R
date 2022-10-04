library(SymSim) # Uses the fork https://github.com/justjhong/SymSim to expose DiscreteEVF
library(ape)
library(anndata)

genCorTree <- function(nTips, cor) {
  root <- stree(1, tip.label = "branching_point")
  tip <- stree(nTips, tip.label = sprintf("CT%i:1", 1:nTips))
  tree <- bind.tree(root, tip, where = 1)
  tree$edge.length <- c(cor, rep(1 - cor, nTips))
  return(tree)
}

genMetaEVFs <- function(n_cells_total, nCellTypes, nMetadata, metadataCor, n_nd_evf, nEVFsCellType, nEVFsPerMetadata, save_dir = NULL, randseed = 1) {
  set.seed(randseed)
  seed <- sample(c(1:1e5), size = nMetadata + 2)

  if (length(metadataCor == 1)) {
    metadataCor <- rep(metadataCor, nMetadata)
  }
  if (length(nEVFsPerMetadata == 1)) {
    nEVFsPerMetadata <- rep(nEVFsPerMetadata, nMetadata)
  }

  ct_tree <- genCorTree(nCellTypes, 0)
  base_evf_res <- DiscreteEVF(ct_tree, n_cells_total, n_cells_total / nCellTypes, 1, 0.4, n_nd_evf, nEVFsCellType, "all", 1, seed[[1]])
  evf_mtxs <- base_evf_res[[1]]
  base_evf_mtx_ncols <- c()
  for (j in 1:3) {
    colnames(evf_mtxs[[j]]) <- paste(colnames(evf_mtxs[[j]]), "base", sep = "_")
    base_evf_mtx_ncols <- c(base_evf_mtx_ncols, ncol(evf_mtxs[[j]]))
  }

  ct_mapping <- ct_tree$tip.label
  names(ct_mapping) <- seq_len(length(ct_mapping))
  meta <- data.frame("celltype" = ct_mapping[base_evf_res[[2]]$pop])

  for (i in 1:nMetadata) {
    shuffled_row_idxs <- sample(1:n_cells_total)
    meta_tree <- genCorTree(2, metadataCor[[i]])
    meta_evf_res <- DiscreteEVF(meta_tree, n_cells_total, n_cells_total / 2, 1, 1.0, 0, nEVFsPerMetadata[[i]], "all", 1, seed[[i + 2]])
    meta_evf_mtxs <- meta_evf_res[[1]]
    for (j in 1:3) {
      colnames(meta_evf_mtxs[[j]]) <- paste(colnames(meta_evf_mtxs[[j]]), sprintf("meta_%d", i), sep = "_")
      evf_mtxs[[j]] <- cbind(evf_mtxs[[j]], meta_evf_mtxs[[j]][shuffled_row_idxs, ])
    }

    meta_mapping <- meta_tree$tip.label
    names(meta_mapping) <- seq_len(length(meta_mapping))
    meta[sprintf("meta_%d", i)] <- meta_mapping[meta_evf_res[[2]]$pop][shuffled_row_idxs]
  }


  # random meta_evfs for cell type 2
  nc_tree <- genCorTree(1, 0)
  nc_evf_res <- DiscreteEVF(nc_tree, n_cells_total / 2, n_cells_total / 2, 1, 1.0, 0, sum(nEVFsPerMetadata), "all", 1, seed[[2]])
  nc_evf_mtxs <- nc_evf_res[[1]]
  ct2_idxs <- which(meta$celltype == "CT2:1")
  for (j in 1:3) {
    evf_mtxs[[j]][ct2_idxs, (base_evf_mtx_ncols[[j]] + 1):ncol(evf_mtxs[[j]])] = nc_evf_mtxs[[j]]
  }

  return(list(evf_mtxs, meta, nc_evf_mtxs))
}


MetaSim <- function(nMetadata, metadataCor, nEVFsPerMetadata, nEVFsCellType, write = F, save_path = NULL, randseed = 1) {
  ncells <- 20000
  ngenes <- 2000
  data(gene_len_pool)
  gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)

  evf_res <- genMetaEVFs(n_cells_total = ncells, nCellTypes = 2, nMetadata = nMetadata, metadataCor = metadataCor, n_nd_evf = 60, nEVFsCellType = nEVFsCellType, nEVFsPerMetadata = nEVFsPerMetadata, randseed = randseed)

  print("simulating true counts")
  true_counts_res <- SimulateTrueCountsFromEVF(evf_res, ngenes = ngenes, randseed = randseed)
  rm(evf_res)
  gc()

  print("simulating observed counts")
  observed_counts <- True2ObservedCounts(true_counts = true_counts_res[[1]], meta_cell = true_counts_res[[3]], protocol = "UMI", alpha_mean = 0.05, alpha_sd = 0.02, gene_len = gene_len, depth_mean = 5e4, depth_sd = 3e3)
  rm(true_counts_res)
  gc()

  print("simulating batch effects")
  observed_counts_2batches <- DivideBatches(observed_counts_res = observed_counts, nbatch = 2, batch_effect_size = 1)
  rm(observed_counts)
  gc()

  print("converting to anndata")
  meta_keys <- c("celltype", "batch")
  meta_keys <- c(meta_keys, paste("meta_", 1:nMetadata, sep = ""))
  ad_results <- AnnData(X = t(observed_counts_2batches$counts), obs = observed_counts_2batches$cell_meta[meta_keys])

  if (write == T && !is.null(save_path)) {
    write_h5ad(ad_results, save_path)
  }

  return(list(ad_results, evf_res))
}

meta_results <- MetaSim(3, metadataCor = c(0, 0.5, 0.9), nEVFsPerMetadata = 7, nEVFsCellType = 40, write = T, save_path = "3_meta_sim_20k.h5ad", randseed = 126)