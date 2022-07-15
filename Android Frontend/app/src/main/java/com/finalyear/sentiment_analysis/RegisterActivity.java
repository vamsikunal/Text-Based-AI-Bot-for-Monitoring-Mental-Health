package com.finalyear.sentiment_analysis;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
//import android.support.v7.app.AppCompatActivity;
import android.util.Patterns;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.FirebaseDatabase;

import java.nio.file.Path;

//import androidx.appcompat.app.AppCompatActivity;


public class RegisterActivity extends AppCompatActivity {

    private FirebaseAuth mAuth;
    private  Button bt;
    private TextView name,email,number,password,nameps,emailps,mobileps,nameec,emailec,mobileec;
    private CardView cv;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);
        mAuth = FirebaseAuth.getInstance();
        changeStatusBarColor();
        bt = (Button) findViewById(R.id.cirNextButton);
        bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                registeruser();
            }
        });
        name = findViewById(R.id.editTextName);
        email = findViewById(R.id.editTextEmail);
        number = findViewById(R.id.editTextMobile);
        password = findViewById(R.id.editTextPassword);
        nameps = findViewById(R.id.editTextNameps);
        emailps = findViewById(R.id.editTextEmailps);
        mobileps = findViewById(R.id.editTextMobileps);
        nameec = findViewById(R.id.editTextNameec);
        emailec = findViewById(R.id.editTextEmailec);
        mobileec = findViewById(R.id.editTextMobileec);
        cv = findViewById(R.id.cv1);

    }
    public void next()
    {
        cv.clearFocus();
    }

    private void registeruser() {
        String naam = name.getText().toString().trim();
        String emailid = email.getText().toString().trim();
        String mobile = number.getText().toString().trim();
        String pass = password.getText().toString().trim();
        String naps = nameps.getText().toString().trim();
        String eps = emailps.getText().toString().trim();
        String mps = mobileps.getText().toString().trim();
        String naec = nameec.getText().toString().trim();
        String eec = emailec.getText().toString().trim();
        String mec = mobileec.getText().toString().trim();
        if(naam.isEmpty())
        {
            name.setError("Please Enter Name!");
            name.requestFocus();
        }
        if(emailid.isEmpty())
        {
            email.setError("Please Enter Email Id!");
            email.requestFocus();

        }
        if(!Patterns.EMAIL_ADDRESS.matcher(emailid).matches())
        {
            email.setError("Please Provide Valid Email Id!");
            email.requestFocus();

        }
        if(mobile.isEmpty() )
        {
            number.setError("Please Enter Mobile Number!");
            number.requestFocus();
        }
        if(mobile.length() != 10)
        {
            number.setError("Please Provide Valid Mobile Number!");
            number.requestFocus();
        }
        if(pass.isEmpty())
        {
            password.setError("Please Enter Password!");
            password.requestFocus();
        }
        if(pass.length()<10)
        {
            password.setError("Minimum Password Length Must be 10!");
            password.requestFocus();
        }
        if(naps.isEmpty())
        {
            nameps.setError("Please Enter Name!");
            nameps.requestFocus();
        }
        if(naec.isEmpty())
        {
            nameec.setError("Please Enter Name!");
            nameec.requestFocus();
        }
        if(eps.isEmpty())
        {
            emailps.setError("Please Enter Email Id!");
            emailps.requestFocus();
        }
        if(eec.isEmpty())
        {
            emailec.setError("Please Enter Email Id!");
            emailps.requestFocus();
        }
        if( !Patterns.EMAIL_ADDRESS.matcher(eps).matches())
        {
            emailps.setError("Please Provide Valid Email Id!");
            emailps.requestFocus();
        }
        if(!Patterns.EMAIL_ADDRESS.matcher(eec).matches())
        {
            emailec.setError("Please Provide Valid Email Id!");
            emailec.requestFocus();
        }
        if(mps.isEmpty())
        {
            mobileps.setError("Please Enter Mobile Number!");
            mobileps.requestFocus();
        }
        if(mec.isEmpty())
        {
            mobileec.setError("Please Enter Mobile Number!");
            mobileec.requestFocus();
        }
        if(mps.length() != 10)
        {
            mobileps.setError("Please Provide Valid Mobile Number!");
            mobileps.requestFocus();
        }
        if(mec.length() != 10)
        {
            mobileec.setError("Please Provide Valid Mobile Number!");
            mobileec.requestFocus();
        }
        else
        {
            next();
        }
        mAuth.createUserWithEmailAndPassword(emailid,pass)
                .addOnCompleteListener(new OnCompleteListener<AuthResult>() {
                    @Override
                    public void onComplete(@NonNull Task<AuthResult> task) {
                        if(task.isSuccessful())
                        {
                            User user = new User(naam,emailid,mobile,naps,eps,mps,naec,eec,mec);
                            FirebaseDatabase.getInstance().getReference("Users")
                                    .child(FirebaseAuth.getInstance().getCurrentUser().getUid())
                                    .setValue(user).addOnCompleteListener(new OnCompleteListener<Void>() {
                                @Override
                                public void onComplete(@NonNull Task<Void> task) {
                                    if(task.isSuccessful())
                                    {
                                        Toast.makeText(RegisterActivity.this,"User Has Registered Succefully!",Toast.LENGTH_LONG).show();
                                    }
                                    else
                                    {
                                        Toast.makeText(RegisterActivity.this,"Failed To Register Please try Again!!",Toast.LENGTH_LONG).show();
                                    }
                                }
                            });
                        }
                        else
                        {
                            Toast.makeText(RegisterActivity.this,"Failed To Register Please try Again!!",Toast.LENGTH_LONG).show();
                        }
                    }
                });

    }

    private void changeStatusBarColor() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            Window window = getWindow();
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
//            window.setStatusBarColor(Color.TRANSPARENT);
            window.setStatusBarColor(getResources().getColor(R.color.register_bk_color));
        }
    }


    public void onLoginClick(View view){
        startActivity(new Intent(this,LoginActivity.class));
        overridePendingTransition(R.anim.slide_in_left,android.R.anim.slide_out_right);

    }



}
