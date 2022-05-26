#import statements
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

#setting up Flask and MySQL
app = Flask(__name__) #this has 2 underscores on each side
app.secret_key = 'himynameistreihaveabasketballgametmrwwhereimapointguardigotshoegameandi'
app.config['MYSQL_HOST'] = 'mysql.2122.lakeside-cs.org'
app.config['MYSQL_USER'] = 'student2122'
app.config['MYSQL_PASSWORD'] = 'm545CS42122'
app.config['MYSQL_DB'] = '2122project'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

#Decorator help from https://flask.palletsprojects.com/en/2.0.x/patterns/viewdecorators/
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('vikrambalaji_loggedIn') == None:
            session['vikrambalaji_loggedIn'] = False
        if session.get('vikrambalaji_loggedIn') == False:
            return redirect('login')
        return f(*args, **kwargs)
    return decorated_function

#Have to make this for pages like login and signup that shouldn't allow user to access when logged in
#This is because they are already logged in and shouldn't access login, signup, first home page
def login_not_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('vikrambalaji_loggedIn') == None:
            session['vikrambalaji_loggedIn'] = False
        if session.get('vikrambalaji_loggedIn') == True:
            return redirect('home')
        return f(*args, **kwargs)
    return decorated_function

#Opening page here
@app.route('/')
@login_not_required
def index():
   session['loggedIn'] = False
   return render_template('openingPageAuxo.html')

#Information for login system
@app.route('/login', methods=['GET', 'POST'])
@login_not_required
def login():
       if request.method == "POST":
           #Whether the number of characters in password + username is odd vs even determines entry or denial
           username = request.form.get('username')
           password = request.form.get('password')

           #Checks whether login credentials (username, password) are valid
           cursor = mysql.connection.cursor()
           query = 'SELECT password FROM vikrambalaji_login2 WHERE username = %s'
           cursor.execute(query, (username, ))
           mysql.connection.commit()
           results = cursor.fetchall()

           if (len(results) == 1):
               hashedPassword = results[0]['password']
               if check_password_hash(hashedPassword, password):
                   session['vikrambalaji_username'] = username
                   session['vikrambalaji_loggedIn'] = True
                   return redirect('home')
               else:
                   return redirect('oops')
           else:
               return redirect('oops')
       else:
           return render_template('auxoLogin.html')

@app.route('/signup', methods=['GET','POST'])
@login_not_required
def signup():
    #The various checks for the password requirements
    if request.method == 'POST':
        password1 = request.form.get('Password')
        password2 = request.form.get('Password2')
        hasDigit = False
        name = request.form.get('Name')
        age = request.form.get('Age')
        username = request.form.get('Username')

        #checks whether password has a number
        for char in password1:
            if char.isdigit():
                hasDigit = True
        if (password1 == password2 and len(password1) >= 8 and hasDigit==True and len(name)>0 and len(age)>0 and len(username)>0):
            #creating cursor to query database
            #inserts information into MySQL database
            securedPassword = generate_password_hash(password2)
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('INSERT INTO vikrambalaji_login2 VALUES (%s, %s, %s, %s, %s, DEFAULT, DEFAULT, DEFAULT, DEFAULT, DEFAULT)', (id, name, age, username, securedPassword, ))
            mysql.connection.commit()
            #This is for singular case where user wants to go from signup to information page
            session['vikrambalaji_username'] = username
            session['vikrambalaji_loggedIn'] = True

            return redirect('information')
        else:
            return redirect('oops')
    else:
        return render_template('auxoSignup.html')

@app.route('/information', methods = ['GET', 'POST'])
#It is required for user to at least signup before accessing information page. Thus, login_required decorator is used
@login_required
def information():
    if (request.method == 'POST'):
        weight = int(request.form.get('Weight'))
        minutes = int(request.form.get('ActivityNum'))
        #Calculation of calories
        target = weight * 11
        if (minutes < 9):
            target = target * 1.1
        elif (minutes > 10 and minutes < 35):
            target = target * 1.2
        elif (minutes > 34):
            target = target * 1.3
        if (request.form.get("toggle") == 'false'):
            target = int(target + 500)
        if (request.form.get("toggle") == 'true'):
            target = int(target - 500)

        carb = int(0.5*target/4)
        fat = int(0.25*target/9)
        protein = int(0.2*target/4)
        sugar = int(0.05*target/4)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #Function above fills in half of table and leaves other half blank, this function completes it
        cursor.execute('UPDATE vikrambalaji_login2 SET cal = %s, carb = %s, fat = %s, protein = %s, sugar = %s WHERE username = %s', (target, carb, fat, protein, sugar, session['vikrambalaji_username'], ))
        mysql.connection.commit()
        return render_template("auxoHome.html")
    else:
        return render_template('auxoInformation.html')

@app.route('/home')
@login_required
def home():
    return render_template('auxoHome.html')

#Nutritional page with tabs
@app.route('/nutrition', methods=['GET','POST'])
@login_required
def nutrition():
    cursor = mysql.connection.cursor()
    query = 'SELECT cal, carb, protein, fat, sugar FROM vikrambalaji_login2 WHERE username = %s'
    cursor.execute(query, (session['vikrambalaji_username'], ))
    mysql.connection.commit()
    nutrition = cursor.fetchall()
    return render_template('auxoNutrition.html', nutrition = nutrition)

#All training information here
@app.route('/training')
@login_required
def training():
    return render_template('auxoTraining.html')

#Training program from Lakeside
@app.route('/program')
@login_required
def program():
    return render_template('auxoTrainingProgram.html')

#Shows how to do many exercises that target specific groups
@app.route('/doExercise')
@login_required
def exerciseHow():
    return render_template('auxoHowExercise.html')

#Information for additional resources - e.g. supplements, discipline, Twitter
@app.route('/additional')
@login_required
def additional():
    return render_template('auxoAdditional.html')

#Allows the user to logout
#Avoided complication with closing tab instead of clicking logout - something I would have implemented had I had more time (mentioned in video)
@app.route('/logout')
def logout():
    session.pop('vikrambalaji_displayName', None)
    session.pop('vikrambalaji_displayAge', None)
    session.pop('vikrambalaji_displayUsername', None)
    session.pop('vikrambalaji_loggedIn', False)
    return redirect('.')

#if the user enters invalid signup or login values
@app.route('/oops')
@login_not_required
def oops():
    return render_template('oops.html.j2')
