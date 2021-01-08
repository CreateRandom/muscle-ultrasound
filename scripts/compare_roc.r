# library('tidyverse')

library('pROC')

# specify the folder that contains the results of the baselines.py script
roc_folder = '/home/klux/PycharmProjects/muscle-ultrasound/roc_analysis'
# specify which dataset and split folder to use

dataset_path = 'ESAOTE_6100_test'
dataset_name = gsub('_',' ',dataset_path)
# read all csv files in path
y_true_path = file.path(roc_folder,dataset_path,'y_true.csv')
y_true <- read.csv(y_true_path)

roc_path = roc_path = file.path(roc_folder,dataset_path,'proba','in_domain')

preds <- list.files(roc_path,pattern='*.csv',full.names = TRUE)
rocs <- list()
names <- list.files(roc_path,pattern='*.csv',full.names = FALSE)
y_preds <- list()

for (i in 1:length(preds)){
  y_pred <- read.csv(preds[i])
  roc_obj = roc(y_true$true, y_pred$pred)
  print(names[i])
  print(coords(roc_obj, "best", input=c("threshold", "specificity","sensitivity"), 
         ret=c("threshold", "specificity", "sensitivity"),as.list=FALSE, drop=TRUE, 
         best.method=c("youden"),best.weights=c(1, 0.5), transpose = FALSE, as.matrix=FALSE))  
  print(max(y_pred$pred))
  print(min(y_pred$pred))
  rocs[[i]] <- roc_obj
  y_preds[[i]] <- y_pred$pred
}
# Test --------------
tests = list()
ps = c()
baseline_name = 'rulebased_original.csv'
baseline_ind = which(names==baseline_name)
for (i in 1:length(rocs)){
  # compare ROCs to baseline
  if(i == baseline_ind){
    next
  }
  else{
    test_result = roc.test(rocs[[i]], rocs[[baseline_ind]])
    tests[[i]] = test_result
    ps[i] = test_result$p.value
  }
}

adjusted_ps = p.adjust(ps)
# https://www.datamentor.io/r-programming/color/
palette = rainbow(length(rocs))

palette[baseline_ind] = 'black'

roc1 <- plot.roc(rocs[[1]], main=dataset_name, percent=TRUE, col=palette[1],asp=1,lwd=3, cex.axis=1.2, cex.lab=1.2)
# iterate over all the remaining ROCs
for (i in 2:length(rocs)){
  plot.roc(rocs[[i]], y_pred$pred, percent=TRUE, col=palette[i],add=TRUE,asp=1,lwd=3,cex.axis=1.2, cex.lab=1.2)
}

# testobj = roc.test(rocs[[1]],rocs[[2]])

extract_name <- function(input_str) {
  stripped_ext = strsplit(input_str,'.csv')[[1]] 
  short = strsplit(stripped_ext,'-')[[1]][1]
  return(short)
}
short_names = sapply(names, extract_name)
short_names[baseline_ind] ='Rule-based method'

legend("bottomright", legend=short_names, col=palette, lwd=3, pt.cex = 1.5)
# text(50, 50, labels=paste("p-value =", format.pval(testobj$p.value)), adj=c(0, .5))



get_threshold <-function(roc_obj,sens=NULL,spec=NULL){
  if(missing(sens) & missing(spec)){
    stop('need to specify either sensitivity or specificity')
  }
  if(missing(sens)){
    target = 'specificities'
    opposite = 'sensitivities'
    value = spec
  }
  else if(missing(spec)){
    target = 'sensitivities'
    opposite = 'specificities'
    value = sens
  }
  opp_values = unlist(roc_obj[opposite],use.names=FALSE)
  values = unlist(roc_obj[target],use.names=FALSE)
  thresholds = unlist(roc_obj['thresholds'],use.names=FALSE)
  operating_ind = which.min(abs(values - value))
  threshold = thresholds[operating_ind]
  opp_value = opp_values[operating_ind]
  returnValues <- list("threshold" = threshold, "best_obtained" = opp_value)
  return(returnValues)
}

# find the best method
aucs = sapply(rocs, function(x) x$auc)

compare_ind = which.max(aucs)
roc_to_compare = rocs[[compare_ind]]
# get baseline performance
baseline = rocs[[baseline_ind]]
specs = unlist(baseline['specificities'],use.names=FALSE)
senses = unlist(baseline['sensitivities'],use.names=FALSE)
max_spec = length(specs) -1
best_spec = specs[max_spec]
sens_obtained = senses[max_spec]
sprintf('Baseline: Best specificity: %s, sensitivity at that threshold: %s', best_spec, sens_obtained)
perfObj = get_threshold(roc_to_compare, spec=best_spec)
sprintf('Method: Obtains %s in sensitivity at same specificity.', perfObj$best_obtained)

max_sens = 2
spec_obtained = specs[max_sens]
best_sens = senses[max_sens]

sprintf('Baseline: Best sensitivity: %s, specificity at that threshold: %s', best_sens, spec_obtained)
perfObj = get_threshold(roc_to_compare, sens=best_sens)
sprintf('Method: Obtains %s in specificity at same sensitivity', perfObj$best_obtained)


# make a plot with cis
roc1 <- plot.roc(rocs[[baseline_ind]], main=dataset_name, percent=TRUE, col=palette[baseline_ind])
plot(roc1)
#Thresholds
ci.thresholds.obj <- ci.thresholds(roc1)
#plot(ci.thresholds.obj)
roc2 <- plot.roc(rocs[[compare_ind]], main=dataset_name, percent=TRUE, col=palette[compare_ind],add =TRUE)
ci.thresholds.obj <- ci.thresholds(roc2)
plot(ci.thresholds.obj)
#plot(roc2)