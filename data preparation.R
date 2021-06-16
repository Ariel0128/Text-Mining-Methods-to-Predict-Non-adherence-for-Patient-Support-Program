#get feelings of the data
table(enroll$StatusReason)
table(haha$CurrentStatus)
table(enroll$AgeRange)
table(enroll$EnrolmentMethod)
table(enroll$JourneyName)
table(enroll$JourneyStatus)

patientnotes <- read.csv(file = 'ptnotes.csv')
enroll <- read.csv(file = 'enroll_c.csv')

notes_lastdate <- sqldf("select enroll.pmid, enroll.LatestStatusChange, enroll.CurrentStatus,
                         patientnotes.[Note.Created.On], 
                         patientnotes.Description, 
                         patientnotes.[Note.Category], 
                         patientnotes.AEEventId
                         from patientnotes 
                         join enroll on patientnotes.pmid=enroll.pmid ")

length(unique(notes_lastdate$pmid))
#214<223, indicates someone has no notes.

write.csv(notes_lastdate,file = "ceased_note.csv")

ptnotes_cc <- sqldf("select enroll.pmid, enroll.LatestStatusChange, enroll.CurrentStatus,
+                          patientnotes.[Note.Created.On], 
+                          patientnotes.Description, 
+                          patientnotes.[Note.Category], 
+                          patientnotes.AEEventId
+                          from enroll 
+                          left join patientnotes on patientnotes.pmid=enroll.pmid ")

#read original dataset

ptnote_org <- read.csv(file = 'C:/Users/Yufan.Wang/Desktop/Data/Touchpoint/patientnotes.csv')

#patient current status is ceased. find her notes to get some ideas.
library(sqldf)
names(patientnotes)[1] <- "pmid"
sqldf("select Description from patientnotes where pmid='TALPAT000021' ")

sqldf("select AEEventDescription from ae where PatientMemberId ='TALPAT000013' ")

#feel the ceased pts
ceased <- sqldf("select StatusReason from enroll where CurrentStatus = 'Ceased' ")
table(ceased)

#PatientMemberID are not exactly same in tables
names(enroll)[4] <- "pmid"
names(interventions)[4] <- "pmid"
names(ae)[5] <- "pmid"

#Join codes - do it later
join_ptnt <- sqldf("select patientnotes.pmid, enroll.PatientId, patientnotes.[Note.Created.On] 
      from enroll
      left join patientnotes on patientnotes.pmid= enroll.pmid ")

#----------------------------------------------

#data cleaning/filtering: remove those unuseful features
write.csv(enroll,file = "filename")
enroll <- sqldf("select * from enroll where CurrentStatus= 'Ceased' or CurrentStatus= 'Complete' or CurrentStatus= 'Enrolled'")
enroll_cc <- sqldf("select * from enroll where CurrentStatus= 'Ceased' or CurrentStatus= 'Complete' ")
interventions_call <- sqldf("select * from interventions where InteractionType= 'Contact Centre' or InteractionType= 'NULL' ")

#text join - concat all the text of the same pt
textjoin <- sqldf("select pmid, group_concat(Description) as ptnote from patientnotes group by pmid")
#join the label/class to the table above
textjoin <- sqldf("select enroll.pmid, textjoin.ptnote, enroll.CurrentStatus
                  from enroll
                  left join textjoin on textjoin.pmid=enroll.pmid")

write.csv(textjoin, file = 'merged_notes.csv')

ae=read.csv('ae_ano.csv')

#duration calculation
enroll$duration<- difftime(enroll$LatestStatusChange ,enroll$EnrolmentDate , units = c("days"))

#half note processing:
enroll<- read.csv('enroll_halfdate.csv')
df1 <- sqldf("select enroll.pmid, enroll.HalfDate, ptnote.[Note.Created.On], ptnote.Description
              from enroll
              left join ptnote on ptnote.pmid=enroll.pmid")
df2 <- sqldf("select pmid, Description from df1 where HalfDate>[Note.Created.On]")
df3 <- sqldf("select pmid, group_concat(Description) as ptnote from df2 group by pmid")
df4 <- sqldf("select enroll.pmid, df3.ptnote, enroll.CurrentStatus
                   from enroll
                   left join df3 on df3.pmid=enroll.pmid")
# for select all from one table in a join 
note1 <- sqldf("select note.*, enroll.EnrolmentDate from note join enroll on enroll.pmid=note.pmid")