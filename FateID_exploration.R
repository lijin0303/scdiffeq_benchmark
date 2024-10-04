pacman::p_load(FateID,arrow,glue,dplyr,ggpubr)
import::from(tidyr,unite,separate,gather,spread)
import::from(magrittr,set_colnames,set_rownames,"%<>%")
import::from(tibble,column_to_rownames,rownames_to_column)
##### Experiment with Larry author's data #####
sharedF <- "https://www.dropbox.com/sh/qrs6a8cqu25dfa9"
ExprFile <- "AAAa7wOhA28cIFLOUEG76Htea/E2_gf.csv"
LabelFile <- "AAAuJG-HwuFZsXnzmr7brMF7a/fate_labels.csv"
options(timeout=600)
x <- read_csv_arrow(glue("{sharedF}/{ExprFile}?dl=1"), col_names=FALSE)
y <- read_csv_arrow(glue("{sharedF}/{LabelFile}?dl=1"), col_names=FALSE)
tar <- c(1,2,3,4)
fb  <- fateBias(x, y, tar, z=NULL, minnr=200, minnrh=200)
saveRDS(fb[1],"fatebias_200_200.rds")
outSave <- "."
write_csv_arrow(fb[1],file=glue('{outSave}/fatebias_200_200.csv'))
##### Subset 20k split test-run #####
splitF <- "Python/FirstRun/"
splitN <- 1
x1 <- read_csv_arrow(glue("{splitF}/FirstSplit.csv"), col_names=FALSE)
y1 <- read_csv_arrow(glue("{splitF}/FirstSplit_label.csv"), col_names=T)$Annotation%>%
  as.factor()
l1 <- as.numeric(y1)
fb  <- fateBias(x1, l1, tar = 1:10, z=NULL, minnr=200, minnrh=200)
saveRDS(fb[1]$probs,glue("fatebias_Subset{splitN}.rds"))
##### Initial pass on the runs #####
probMat <- readRDS(glue("fatebias_Subset{splitN}.rds"))
metaSplit <- read.delim("Python/FirstSplit_Meta.csv",sep=",")[,-c(1,7)]
majorVoteList <- apply(probMat,1,function(s) levels(y1)[-11][s>0.5])
majorVoteList[lengths(majorVoteList)==0] <- "NotDetermined"
metaSplit%<>%mutate(FateBias_Major = unlist(majorVoteList))
# First Sanity check, for target cluster
metaSplit%>%filter(Annotation!="undiff" & Annotation!=FateBias_Major)
metaSplit%>%filter(Annotation=="undiff")%>%group_by(FateBias_Major) %>% summarise(n=n())
# Find closest neighbors

##### Trying to subset and run all =====
options(timeout=600)
x_whole <- read_csv_arrow("Data/WholeExpr.csv", col_names=FALSE)
meta_whole <- read_csv_arrow("Data/WholeMeta.csv", col_names=T)
indexFile <- "Data/Subset_2k_Seed123.csv"
for (i in 2:4){
  message(glue("Random Sample 2k Batch {i}!"))
  indexRun <- as.integer(read.csv(indexFile, skip=i, nrows=1)[,-1])
  SubMat  <- x_whole[meta_whole$cell_idx %in% indexRun,]
  SubExpr <- as.data.frame(t(SubMat))
  colnames(SubExpr) <- paste0("c",1:ncol(SubExpr))
  SubMeta <- meta_whole[meta_whole$cell_idx %in% indexRun,]
  SubAnnot <- factor(SubMeta$Annotation)
  SubLabel <- as.numeric(SubAnnot)
  names(SubLabel) <- colnames(SubExpr)
  fb  <- fateBias(SubExpr, SubLabel, tar = 1:10, z=NULL, minnr=200, minnrh=200)
  out <- fb[1]$probs
  colnames(out) <- levels(SubAnnot)[-11]
  rownames(out) <- indexRun
  saveRDS(out,glue("fatebias_SubsetSplit{i}.rds"))
}
