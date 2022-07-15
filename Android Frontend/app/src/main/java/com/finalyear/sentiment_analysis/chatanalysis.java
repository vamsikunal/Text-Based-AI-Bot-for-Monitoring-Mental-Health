package com.finalyear.sentiment_analysis;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.squareup.picasso.Picasso;

import org.jetbrains.annotations.NotNull;
import org.w3c.dom.Text;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.MediaType;
import okhttp3.OkHttp;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class chatanalysis extends AppCompatActivity {

    private Button bt;
    private TextView tv;
    private OkHttpClient mclient;
    private ImageView img;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chatanalysis);
        changeStatusBarColor();
        mclient = new OkHttpClient();
        Request request = new Request.Builder().url("http://192.168.1.117:5000/").build();
        tv = findViewById(R.id.textView8);
                mclient.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(@NotNull Call call, @NotNull IOException e) {
                        Log.e(">> error", "Error");
                    }

                    @Override
                    public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                        if(response.isSuccessful())
                        {

                            final String str = response.body().string();
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    tv.setText(""+str);
                                    img = findViewById(R.id.imageView2);
                                    Picasso.get()
                                            .load("http://192.168.1.117:5000/test")
                                            .into(img);
                                }
                            });

                        }
                    }
                });
                img = findViewById(R.id.imageView2);


            }
    private void changeStatusBarColor() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            Window window = getWindow();
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
            //window.setStatusBarColor(Color.TRANSPARENT);
            window.setStatusBarColor(getResources().getColor(R.color.register_bk_color));
        }
    }
}

