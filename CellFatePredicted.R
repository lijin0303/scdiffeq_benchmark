setwd("~/Rotation/Gaddy_Rotation/mv_scDiffEq/")
CorTruth <- function(rdsf){
  TrueSet <- read_csv_arrow("Data/GrounTruth.csv")
  testKNN_whole <- read_csv_arrow("Data/KNN_testSet.csv", col_names=FALSE)%>% 
    filter(f0 %in% TrueSet$cell_idx)
  FBmat <- readRDS(rdsf)
  KNNInt <- pbapply::pbapply(testKNN_whole,1,function(s) length(intersect(s[-1],rownames(FBmat))),cl=10)
  testKNN_Sub <- testKNN_whole[KNNInt!=0,]
  MeanFB <- pbapply::pblapply(seq(nrow(testKNN_Sub)),function(i){apply(FBmat[rownames(FBmat) %in% testKNN_Sub[i,],],2,mean)},cl=10)
  CellFB_test <- bind_rows(MeanFB) %>% 
    mutate(cell_idx  = testKNN_Sub$f0)
  CellFateSub <- CellFB_test %>% 
    inner_join(TrueSet%>%select(cell_idx,TrueRatio = neu_vs_mo_percent))
  CellFateSub%<>%
    mutate(PredictedRatio = Monocyte/(Monocyte+Neutrophil))%>%
    select(cell_idx,TrueRatio,PredictedRatio)
  o <- cor.test(CellFateSub$TrueRatio,CellFateSub$PredictedRatio)
  return(data.frame(t = o$statistic,rho = o$estimate,pval = o$p.value))
}
AllCors <- lapply(0:4,function(i) CorTruth(glue("fatebias_SubsetSplit{i}.rds")))
TabCors <- bind_rows(AllCors)%>%
  mutate_all(round,digits=3)
tab <- ggtexttable(TabCors,rows = NULL,
                   theme = ttheme(
                     colnames.style = colnames_style(color = "black", size = rel(15)),
                     tbody.style = tbody_style(color = "black",fill = c("gray90"), 
                                               size = rel(12))))

require(ggstatsplot)
set.seed(123)
TabCors%<>%mutate(method="FateID")
violin2 <- ggbetweenstats(
  data  = TabCors,
  x     = method,
  y     = rho,
  title = "",
  point.args = list(position = ggplot2::position_jitterdodge(dodge.width = 0.6), alpha
                    = 1, size = 3.5, stroke = 0),
  xlab="",ylab=""
)

ggarrange(tab,violin2,nrow=1,labels = "Pearson Correlation")
ggsave("Correlation_results.pdf",width = 8,height = 5.5)
##### Separate Codes =====
testKNN_whole <- read_csv_arrow("Data/KNN_testSet.csv", col_names=FALSE)
FBmat = out
FBmat <- readRDS("fatebias_SubsetSplit1.rds")
KNNInt <- pbapply::pbapply(testKNN_whole,1,function(s) length(intersect(s[-1],rownames(FBmat))),cl=10)
testKNN_Sub <- testKNN_whole[KNNInt!=0,]
MeanFB <- pbapply::pblapply(seq(nrow(testKNN_Sub)),function(i){apply(FBmat[rownames(FBmat) %in% testKNN_Sub[i,],],2,mean)},cl=10)
CellFB_test <- bind_rows(MeanFB) %>% 
  mutate(cell_idx  = testKNN_Sub$f0)
write_csv_arrow(CellFB_test,"Data/Predicted_CellFate_Test80k.csv")
write_csv_arrow(out%>%rownames_to_column("cell_idx"),"Data/Training_CellFate_Sample2k.csv")
CellFB_test <- read_csv_arrow("Data/Predicted_CellFate_Test80k.csv")
TrueSet <- read_csv_arrow("Data/GrounTruth.csv")
CellFateSub <- CellFB_test %>% 
  inner_join(TrueSet%>%select(cell_idx,TrueRatio = neu_vs_mo_percent))
CellFateSub%<>%
  mutate(PredictedRatio = Neutrophil/Monocyte)%>%
  select(cell_idx,TrueRatio,PredictedRatio)
cor.test(CellFateSub$TrueRatio,CellFateSub$PredictedRatio)

