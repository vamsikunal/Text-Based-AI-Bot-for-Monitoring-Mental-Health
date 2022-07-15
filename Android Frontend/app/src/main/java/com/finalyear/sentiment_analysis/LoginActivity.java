package com.finalyear.sentiment_analysis;

import android.content.Intent;
import android.os.Build;
//import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Patterns;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;


public class LoginActivity extends AppCompatActivity {
    private EditText email,password;
    private Button bt;
    private FirebaseAuth mAuth;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //for changing status bar icon colors
        if(Build.VERSION.SDK_INT>= Build.VERSION_CODES.M){
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        }
        setContentView(R.layout.activity_login);
        email = findViewById(R.id.editTextEmail);
        password = findViewById(R.id.editTextPassword);
        mAuth = FirebaseAuth.getInstance();
        bt = findViewById(R.id.cirLoginButton);
        bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                login();
            }
        });
    }

    private void login() {
        String emailid = email.getText().toString().trim();
        String pass = password.getText().toString().trim();
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
        if(pass.isEmpty())
        {
            password.setError("Please Enter Password!");
            password.requestFocus();
        }
        mAuth.signInWithEmailAndPassword(emailid,pass).addOnCompleteListener(new OnCompleteListener<AuthResult>() {
            @Override
            public void onComplete(@NonNull Task<AuthResult> task) {
                if(task.isSuccessful())
                {
                    startActivity(new Intent(LoginActivity.this,Dashboard.class));
                }
                else
                {
                    Toast.makeText(LoginActivity.this,"Failed To login please check Your Credentials",Toast.LENGTH_LONG).show();
                }
            }
        });
    }

    public void onLoginClick(View View){
        startActivity(new Intent(this,RegisterActivity.class));
        overridePendingTransition(R.anim.slide_in_right,R.anim.stay);

    }
}
