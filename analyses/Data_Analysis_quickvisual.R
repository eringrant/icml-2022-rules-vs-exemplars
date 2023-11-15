library(ggplot2)
require(gridExtra)
library(plyr)
library(dplyr)
library(plotly)
library(ggthemes)
library(tidyr)
library(wesanderson)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")



# setwd("/Users/ishita/GitHub/exposure-bias/local_run_data/toyMLP/MonFeb1_no3lys_15ksteps/")
# df = read.csv('aggregate.csv')
# 
# df = subset(df, x_ffst_bs == 5 & y_ffst_bs == 5)
# df = subset(df, Train_acc == 1)
# df = subset(df, step == 7)
# 
# df$Valid_acc = NULL
# df$Valid_loss = NULL
# 
# head(df)
# 
# summ  = ddply(df, .(n_lrs, expt), summarize, 
#               mean_loss = mean(Test_loss),
#               ci_loss = 1.96*sd(Test_loss)/sqrt(length(Test_loss)),
#               mean_acc = mean(Test_acc),
#               ci_acc = 1.96*sd(Test_acc)/sqrt(length(Test_acc))
#               )
# summ$n_lrs = factor(summ$n_lrs, levels = c(0, 1, 2))
# 
# 
# p_acc <- ggplot(summ, aes(x=expt, y=mean_acc, fill=n_lrs)) + 
#   geom_bar(stat="identity", position=position_dodge()) + ylim(c(-0.1, 1.1))+
#   geom_errorbar(aes(ymin=mean_acc-ci_acc, ymax=mean_acc+ci_acc), width=.1,
#                 position=position_dodge(.9))
# 
# p_acc
# 
# 
# p_loss <- ggplot(summ, aes(x=expt, y=mean_loss, fill=n_lrs)) + 
#   geom_bar(stat="identity", position=position_dodge()) + 
#   geom_errorbar(aes(ymin=mean_loss-ci_loss, ymax=mean_loss+ci_loss), width=.1,
#                 position=position_dodge(.9))
# 
# p_loss

setwd("/Users/ishita/GitHub/exposure-bias/Notebooks/")
# name = 'ICML_final_200runs'
# name = 'axis_misaligned_200runs'
#name = 'all_rho_settings'
# name = 'matched_corr_50runs_withresample'
name = 'linear_all_rho_settings'
df = read.csv(paste(name, '.csv', sep = ''))

# only those that learn
df = subset(df, df$train_perf>0.9)
# remove white space
df$classifier = sapply(df$classifier,function(x)gsub('\\s+', '',x))
df$classifier = factor(df$classifier)
df$expt = sapply(df$expt,function(x)gsub('\\s+', '',x))


model_namesfrom = c("logreg",'MLP_1H_1L', 'MLP_2H_1L',
              'MLP_2H_4L', 'MLP_3H_1L', 'MLP_4H_1L', 
              'MLP_5H_1L',  'MLP_16H_1L', 
              'gauss_kernel', 'lin_kernel',
              'fitted_gauss_kernel', 'gauss_kernel_ls0.1', 'gauss_kernel_ls0.5', 
              'gauss_kernel_ls2', 'gauss_kernel_ls4',
              'gauss_kernel_ls6', 'gauss_kernel_ls8',
              'gauss_kernel_ls10')
model_namesto = c('0H0L', '1H1L', '2H1L', 
                  '2H4L', '3H1L','4H1L', 
                  '5H1L',  '16H1L', 
                  'RBF GP', 'Lin kernel',
                  'fit', 'LS0.1',  'LS0.5',
                  'LS2.0', 'LS4.0', 
                  'LS6.0','LS8.0', 
                  'LS10.0')

df$classifier <- mapvalues(df$classifier, from = model_namesfrom,
                           to = model_namesto)

# cond_namesfrom = c("conflict", 'zero_shot', 'partial_exposure')
# cond_namesto = c('Cue Conflict', 'Zero Shot', 'Partial Exposure')
# cond_namesfrom = c("PE", 'small_dev', 'large_dev')
# cond_namesto = c("Partial Exposure", 'Small Deviation', 'Large Deviation')

df$expt <- factor(df$expt)
# df$expt <- mapvalues(df$expt, from = cond_namesfrom,
#                      to = cond_namesto)

df = subset(df, !(classifier == "Lin kernel"))
summ  = ddply(df, .(classifier, expt), summarize,
              mean = mean(test_perf),
              ci = 1.96*sd(test_perf)/sqrt(length(test_perf))
              )

mlp_summ = subset(summ, !sapply(summ$classifier, function (x) grepl('kernel', x)))

# fix ordering of columns

# mlp_summ$classifier = factor(mlp_summ$classifier, 
#                              levels = model_namesto)
# 
# mlp_summ$expt = factor(mlp_summ$expt, 
                             levels = cond_namesto)
mlp_p <- ggplot(mlp_summ, aes(x=expt, y=mean, fill=classifier)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=mean-ci/2, ymax=mean+ci/2), width=.1,
                position=position_dodge(.9))+
  xlab('Training condition') +
  scale_y_continuous(name ="Test performance", #limits = c(0.0, 1.05),
                   breaks = seq(0.0, 1.0, by = 0.25), expand = c(0, 0)) +
  theme_hc()


mlp_p +scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)
ggsave(paste(name, '_all.pdf', sep = ''))

# only for rbfs 

rbf_summ = subset(summ, sapply(summ$classifier, function (x) grepl('kernel', x)))

rbf_summ$classifier <- mapvalues(rbf_summ$classifier, from = model_namesfrom,
                                 to = model_namesto)

# fix ordering of columns

rbf_summ$classifier = factor(rbf_summ$classifier, 
                             levels = model_namesto)

rbf_summ$expt = factor(rbf_summ$expt, 
                       levels = cond_namesto)
rbf_p <- ggplot(rbf_summ, aes(x=expt, y=mean, fill=classifier)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=mean-ci/2, ymax=mean+ci/2), width=.1,
                position=position_dodge(.9))+
  xlab('Training condition') +
  scale_y_continuous(name ="Test performance", #limits = c(0.0, 1.05),
                     breaks = seq(0.0, 1.0, by = 0.25), expand = c(0, 0)) +
  theme_hc()


rbf_p +scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)
ggsave(paste(name, '_all_rbf.pdf', sep = ''))



###########################
# making difference plots
###########################


df_long = spread(df, expt, test_perf)
#df_long$pe_effect = df_long$`Zero Shot` - df_long$`Partial Exposure`
df_long$pe_effect = df_long$zero_shot - df_long$partial_exposure
#df_long$baseline = 0.5 - df_long$`Cue Conflict`
df_long$`Cue Conflict` = NULL
df_long$`Partial Exposure` = NULL
df_long$`Zero Shot` = NULL
df_long$train_perf = NULL
df_long_lim = subset(df_long, classifier != '2H4L')
df_long_lim = subset(df_long, classifier != 'LS0.1')

summ_m  = ddply(df_long, .(classifier), summarize,
              pe = mean(pe_effect, na.rm = TRUE),
              ci = 1.96*sd(pe_effect, na.rm = TRUE)/sqrt(length(pe_effect[!is.na(pe_effect)])))

# horizontal barplot

summ_m$ci[is.na(summ_m$ci)] <- 0.0001

summ_m$classifier = factor(summ_m$classifier), 
                             #levels = model_namesto)
mlp_p <- ggplot(summ_m, aes(x=classifier, y=pe, fill = classifier)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=pe-ci/2, ymax=pe+ci/2), width=.1,
                position=position_dodge(.9))+
  xlab('Models') + coord_flip() + ylab('Partial Exposure Effect')+
  # scale_y_continuous(name ="Effect size", limits = c(-0.1, 0.5),
  #                    breaks = seq(0.0, 1.0, by = 0.1), expand = c(0, 0)) +
  # 
  scale_fill_brewer(palette="YlOrRd") + theme_minimal() +  
  theme(legend.position="none", text = element_text(size=20)) 


mlp_p 
ggsave(paste(name, '_pe.pdf', sep = ''))

###########################
# depth height barplot
###########################

include = c('2H4L', '16H1L')
df_long_lim = subset(df_long, classifier %in% include)

summ_m  = ddply(df_long_lim, .(classifier), summarize,
                pe = mean(pe_effect, na.rm = TRUE),
                ci = 1.96*sd(pe_effect, na.rm = TRUE)/sqrt(length(pe_effect[!is.na(pe_effect)])))

summ_m$classifier = factor(summ_m$classifier, 
                           levels = model_namesto)
mlp_p <- ggplot(summ_m, aes(x=classifier, y=pe, fill = classifier)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=pe-ci/2, ymax=pe+ci/2), width=.1,
                position=position_dodge(.9))+
  xlab('Models') + ylab('Partial Exposure Effect')+
  # scale_y_continuous(name ="Effect size", limits = c(-0.1, 0.5),
  #                    breaks = seq(0.0, 1.0, by = 0.1), expand = c(0, 0)) +
  # 
  scale_color_manual(values = c("#56B4E9", "#0072B2"))  + theme_minimal() +  
  theme(legend.position="none", text = element_text(size=20)) 



mlp_p 
ggsave(paste(name, '_hw.pdf', sep = ''))
# scale_fill_brewer(palette="YlGn")


# 
# p <- ggplot(example.df, aes(y = Author, x = d, xmin = ci.low, xmax = ci.high, shape=Gender)) +
#   geom_point() +
#   geom_errorbarh(height = .1) +
#   scale_x_continuous(limits=c(-2,2),breaks=c(-2,-1.5,-1,-0.5,0,.5,1,1.5,2))+
#   geom_vline(xintercept=0, color="grey60",linetype="dashed")+
# mlp_p
# 
# 
# summ_ci  = ddply(df_long, .(classifier), summarize,
#                pe = 1.96*sd(pe_effect, na.rm = TRUE)/sqrt(length(pe_effect[!is.na(pe_effect)])),
#                base = 1.96*sd(baseline, na.rm = TRUE)/sqrt(length(baseline[!is.na(baseline)])))
# summ_m = gather(summ_m, condition, mean, c(pe,base), factor_key=TRUE)
# summ_ci = gather(summ_ci, condition, ci, c(pe, base), factor_key=TRUE)
# 
# summ = merge(summ_ci, summ_m)
# 
# summ$classifier = factor(summ$classifier, 
#                              levels = model_namesto)
# gp_summ = subset(summ, sapply(summ$classifier, function (x) grepl('kernel', x)))
# 
# gp_p <- ggplot(gp_summ, aes(x=condition, y=mean, fill=classifier)) +
#   geom_bar(stat="identity", position=position_dodge()) +
#   geom_errorbar(aes(ymin=mean-ci/2, ymax=mean+ci/2), width=.1,
#                 position=position_dodge(.9))+
#   xlab('Training condition') +
#   scale_y_continuous(name ="Test performance", limits = c(-0.1, 0.5),
#                      breaks = seq(0.0, 1.0, by = 0.2), expand = c(0, 0)) +
#   theme_hc() 
# gp_p
# 
# 
# mlp_summ = subset(summ, !sapply(summ$classifier, function (x) grepl('kernel', x)))
# 
# mlp_p <- ggplot(mlp_summ, aes(x=condition, y=mean, fill=classifier)) +
#   geom_bar(stat="identity", position=position_dodge()) +
#   geom_errorbar(aes(ymin=mean-ci/2, ymax=mean+ci/2), width=.1,
#                 position=position_dodge(.9))+
#   xlab('Differences') +
#   scale_y_continuous(name ="Effect size", limits = c(-0.1, 0.36),
#                      breaks = seq(0.0, 1.0, by = 0.1), expand = c(0, 0)) +
#   theme_hc()  + coord_flip()
# mlp_p +scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)
# 
