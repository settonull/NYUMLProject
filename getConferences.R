
library(RCurl)
library(XML)
library(tidyverse)

setwd('C:/Users/isaac/OneDrive/Documents/NYU/Spring_2018/c_1003/NYUMLProject')

getTeams <- function(conf, year){
  conf <- as.character(conf)
  
  
  conf2 <- ifelse((year < 2011)&(grepl('pac-12',conf)), 'pac-10', conf)
  cat(conf2, '-', year, '\n') 
  
  link <- paste0('https://www.sports-reference.com/cfb/conferences/', conf2, '/', year, '.html')
  
  confNames <- c('Team', 'W', 'L', 'Pct', 
                 'InW', 'InL', 'InPct', 
                 'PtsF', 'PtsA', 'SRS', 'SOS', 'APPre', 'APHigh', 'APPost', 'Notes')
  confYear <- readHTMLTable(getURL(link), as.data.frame=TRUE)[[1]] %>% 
    setNames(., confNames) %>% 
    as.tibble() %>% 
    filter(!grepl('W', W, fixed=TRUE)) %>%
    select(one_of('Team')) %>%
    mutate(Year=year, Conf=conf)
  
  return(confYear)
  
}

webNames <- c('acc','american','big-12','big-ten','cusa','independent','mac','mwc','pac-12','sec','sun-belt','big-east','wac')
colNames <- c('Rk','Conf','From','To','W','L','T','Pct','InW','InL','InT','InPct','SRS','SOS','AP','Notes')
link <- 'https://www.sports-reference.com/cfb/conferences/'
data <- readHTMLTable(getURL(link), as.data.frame=TRUE)[[1]] %>% 
  setNames(., colNames) %>% 
  as.tibble() %>%
  select(one_of('Conf', 'From', 'To')) %>%
  filter(!grepl('Overall', Conf, fixed=TRUE)) %>% 
  filter(!grepl('From', From, fixed=TRUE)) %>%
  mutate(From = as.numeric(as.character(From)), To = as.numeric(as.character(To))) %>%
  filter(To > 2001) %>% 
  mutate(Web_id = webNames) %>% 
  rowwise %>% mutate(From = max(2001, From))


complete <- apply(data, 1, function(x) rev(lapply(x['From']:x['To'], function(year) getTeams(x['Web_id'], year))))

do.call('rbind', unlist(complete, recursive = FALSE)) %>% write.csv(., 'data/complete_conf.csv', row.names=FALSE)
  



