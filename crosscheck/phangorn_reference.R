# Reference values from ape/phangorn for the datasets written by run_crosscheck.py.
# This script is only used to check the Python implementation; nothing in the
# package depends on R.
#
# Usage: Rscript crosscheck/phangorn_reference.R <work_dir>
# Needs the R packages ape and phangorn (install.packages("phangorn")).

suppressPackageStartupMessages({
  library(ape)
  library(phangorn)
})

args <- commandArgs(trailingOnly = TRUE)
work <- if (length(args) > 0) args[1] else "crosscheck/work"
datasets <- sort(sub("\\.fasta$", "", list.files(work, pattern = "\\.fasta$")))
rows <- list()

for (ds in datasets) {
  fasta <- file.path(work, paste0(ds, ".fasta"))
  dna <- read.dna(fasta, format = "fasta")
  dat <- phyDat(dna, type = "DNA")

  # distances, UPGMA and NJ
  d <- dist.dna(dna, model = "JC69", pairwise.deletion = TRUE)
  write.csv(as.matrix(d), file.path(work, paste0(ds, "_R_dist.csv")))
  write.tree(phangorn::upgma(d), file.path(work, paste0(ds, "_R_upgma.nwk")))
  write.tree(ape::nj(d), file.path(work, paste0(ds, "_R_nj.nwk")))

  # score the Python trees with phangorn
  for (method in c("upgma", "nj", "parsimony", "ml", "true")) {
    f <- file.path(work, paste0(ds, "_py_", method, ".nwk"))
    if (!file.exists(f)) next
    tr <- unroot(read.tree(f))
    tr$edge.length[tr$edge.length < 0] <- 0
    fit <- pml(tr, dat, model = "JC")
    fit_opt <- optim.pml(fit, optNni = FALSE, optEdge = TRUE, control = pml.control(trace = 0))
    rows[[length(rows) + 1]] <- data.frame(
      dataset = ds, tree = method,
      R_parsimony = parsimony(tr, dat, method = "fitch"),
      R_loglik_given = fit$logLik,
      R_loglik_opt_edges = fit_opt$logLik
    )
  }

  # phangorn's own searches, for comparison with ours
  start <- ape::nj(dist.ml(dat, model = "JC69"))
  ml <- optim.pml(pml(start, dat, model = "JC"), optNni = TRUE, optEdge = TRUE,
                  control = pml.control(trace = 0))
  write.tree(ml$tree, file.path(work, paste0(ds, "_R_ml.nwk")))
  set.seed(1)
  mp <- pratchet(dat, trace = 0, minit = 50)
  write.tree(mp, file.path(work, paste0(ds, "_R_parsimony.nwk")))
  rows[[length(rows) + 1]] <- data.frame(
    dataset = ds, tree = "R_search",
    R_parsimony = parsimony(mp, dat, method = "fitch"),
    R_loglik_given = NA, R_loglik_opt_edges = ml$logLik
  )
}

out <- do.call(rbind, rows)
write.csv(out, file.path(work, "R_scores.csv"), row.names = FALSE)
cat("wrote", file.path(work, "R_scores.csv"), "\n")
