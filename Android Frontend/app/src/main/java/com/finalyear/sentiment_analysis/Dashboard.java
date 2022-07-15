package com.finalyear.sentiment_analysis;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class Dashboard extends AppCompatActivity {

    private CardView cv,cv1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dashboard);
        cv = findViewById(R.id.cv1);
        cv1 = findViewById(R.id.cv2);
        cv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(Dashboard.this,chatanalysis.class));
                overridePendingTransition(R.anim.slide_in_right,R.anim.stay);
            }
        });
        cv1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(Dashboard.this,groupchat.class));
                overridePendingTransition(R.anim.slide_in_right,R.anim.stay);
            }
        });
    }
}