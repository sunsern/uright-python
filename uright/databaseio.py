import xml.dom.minidom as minidom
import MySQLdb as mdb
import json

def inkxml2jsonObj(xmlStr):
    dom = minidom.parseString(xmlStr.encode("utf-8"))
    character = dom.getElementsByTagName("character")[0]
    baseline = float(character.getElementsByTagName("baseline")[0].
                     childNodes[0].data)
    topline = float(character.getElementsByTagName("topline")[0].
                    childNodes[0].data)
    charDict = {"baseline":baseline, "topline":topline}
    strokes = character.getElementsByTagName("stroke")
    strokeArray = []
    for stroke in strokes:
        points = stroke.getElementsByTagName("point")
        pointArray = []
        for point in points:
            x = float(point.getElementsByTagName("x")[0].childNodes[0].data)
            y = float(point.getElementsByTagName("y")[0].childNodes[0].data)
            t = float(point.getElementsByTagName("t")[0].childNodes[0].data)
            pointDict = {"x":x, "y":y, "t":t}
            pointArray.append(pointDict)
        strokeArray.append(pointArray)
    charDict["strokes"] = strokeArray
    return charDict

class DatabaseIO:
    def __init__(self,server,username,password,dbname):
        self.server = server
        self.username = username
        self.password = password
        self.dbname = dbname


    def retrieveRawInk(self,userid,languageid,max_examples=20):
        # create a new connection to the db
        con = mdb.connect(self.server, self.username,
                          self.password, self.dbname,
                          charset='utf8', use_unicode = True);

        # read available labels
        cur = con.cursor(mdb.cursors.DictCursor)
        cur.execute("SELECT DISTINCT label "
                    "FROM inkdata WHERE "
                    "user_id=%d AND language_id=%d"%(userid,languageid))
        rows = cur.fetchall()
        labels = [ row["label"] for row in rows ]
        
        # read max_examples of each label
        raw_ink = {}
        for c in labels:
            cur.execute("SELECT inkdata.* FROM inkdata "
                        "LEFT JOIN sessions "
                        "ON inkdata.session_id = sessions.session_id "
                        "WHERE "
                        "inkdata.user_id=%d AND inkdata.language_id=%d "
                        "AND inkdata.label=\"%s\" "
                        "AND sessions.mode_id=3 " 
                        "ORDER BY ink_id DESC "
                        "LIMIT %d" %
                        (userid, languageid, c, max_examples))
            rows = cur.fetchall()
            # create json object from each example
            for row in rows:
                # backward compatible
                if (row["ink"].startswith("<character>")):
                    example = _inkxml2jsonObj(row["ink"])
                else:
                    example = json.loads(row["ink"])
                raw_ink.setdefault(c,[]).append(example)
                
        # close the connnection
        con.close()
        return raw_ink

def dumpRawInk(langid=1, max_examples=30):
    import pickle
    import time
    
    users = [1,6,22,9,29,32,35,43,50]
    dbio = DatabaseIO('localhost','scheaman',
                      'sunsern', 'ios_experiments')
    all_ink = {}
    print "retrieving ink..."
    for userid in users:
        print " > user %d"%userid
        rink = dbio.retrieveRawInk(userid,langid,max_examples=max_examples)
        if len(rink.items()) > 0:
            all_ink["user_%d"%userid] = rink

    o_filename = "rawink_%d_%0.0f.p"%(langid,time.time())
    pickle.dump(all_ink, open(o_filename,"wb"))

def _test():
    dbio = DatabaseIO('localhost', 'scheaman',
                      'sunsern', 'ios_experiments_2')
    raw_ink = dbio.retrieveRawInk(35,1,max_examples=10)
    print "# labels =",len(raw_ink.keys())
    print "# ink for each label =",len(raw_ink[raw_ink.keys()[0]])

if __name__ == '__main__':
    _test()
        
