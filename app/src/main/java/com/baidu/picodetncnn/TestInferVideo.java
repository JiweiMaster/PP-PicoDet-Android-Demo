package com.baidu.picodetncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.DecimalFormat;

public class TestInferVideo extends Activity {
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    Button testDelayBtn;
    Button jumpMainBtn;
    NanoDetNcnn nanoDetNcnn = new NanoDetNcnn();
    private int current_model = 0;
    private int current_cpugpu = 0;
    TextView inferTimeTextView;
    ImageView inferImageView;
    LinearLayout inferLayout;
    TextView fpsTextView;
    int count = 0;
    boolean isShowImgThread = true;
    Thread showImgThread;
//    int frameCount = 1051;
    int frameCount = 124;
    boolean isRunThread = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test_infer_video);
        testDelayBtn = this.findViewById(R.id.testDelayBtn);
        spinnerModel = (Spinner) findViewById(R.id.spinnerModel2);
        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU2);
        inferTimeTextView = this.findViewById(R.id.inferTime);
        jumpMainBtn = this.findViewById(R.id.jumptInferPageBtn);
        inferImageView = this.findViewById(R.id.showInferImg);
        inferLayout = this.findViewById(R.id.inferLayout);
        fpsTextView = this.findViewById(R.id.fpsTextView);
        System.out.println("ncnn: copyMP42Cache");
        copyMP42Cache();
        testDelayBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isRunThread = true;
                inferTimeTextView.setText("star cal...");
                double inferTime = nanoDetNcnn.testInferTime(getAssets(), current_model, current_cpugpu);
                double fps = frameCount*1000/inferTime;
//                inferTimeTextView.setText("FPS : "+String.valueOf(frameCount*1000/inferTime));
                DecimalFormat decimalFormat = new DecimalFormat("#.00");
                String fpsStr = decimalFormat.format(fps);
                fpsTextView.setText("fps: "+fpsStr);
                getShowImgThread().start();
            }
        });

        jumpMainBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(TestInferVideo.this, MainActivity.class));
            }
        });

        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    System.out.println("current_model => "+current_model);
                    try{
                        isRunThread = false;
                    }catch (Exception e){
                        Log.d("ncnn: ", "stop thread ...");
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });
//        切换cpu/gpu
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    System.out.println("current_model => "+current_model);
                    try{
                        isRunThread = false;
                    }catch (Exception e){
                        Log.d("ncnn: ", "stop thread ...");
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });
    }

    public Thread getShowImgThread(){
        return new Thread(new Runnable() {
            @Override
            public void run() {
                File fileDir = new File(TestInferVideo.this.getExternalCacheDir()+"");
                Log.d("ncnn:","file path =>"+fileDir.getAbsolutePath());
                while(isRunThread){
                    try {
                        String imageName = fileDir.list()[count%frameCount];
                        count ++;
                        if(imageName.contains(".jpg")){
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Bitmap bitmap = BitmapFactory.decodeFile(TestInferVideo.this.getExternalCacheDir() +"/"+imageName);
                                    inferImageView.setImageBitmap(bitmap);
                                }
                            });
                        }
                        Thread.sleep(300);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

            }
        });

    }

    public void copyMP42Cache() {
        String filename = "out.avi";
        File file = new File(TestInferVideo.this.getExternalCacheDir()+"/"+filename);
        Log.d("ncnn: src video dir => ", file.getAbsolutePath());
        if (!file.exists()) {
            try {
                InputStream is = TestInferVideo.this.getAssets().open(filename);
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();
                FileOutputStream fos = new FileOutputStream(file);
                fos.write(buffer);
                fos.close();
            } catch (Exception e) {
                Log.d("ncnn: copy src file to dst file failed => ", file.getAbsolutePath());
                throw new RuntimeException(e);
            }
        } else {
            Log.d("ncnn: dst file exist => ", file.getAbsolutePath());
        }
    }
    @Override
    protected void onResume() {
        super.onResume();
        if (TestInferVideo.this.checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED)
        {
            TestInferVideo.this.requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, 102);
        }
    }
}