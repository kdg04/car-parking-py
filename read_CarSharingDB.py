import csv
import sqlite3

con = sqlite3.connect('CarSharingDB.db')
print("connection created")
cursor = con.cursor()
cursor.execute("SELECT * FROM Car_sharing")
rows = cursor.fetchall()
for row in rows:
    print(row)
con.commit()
con.close()

