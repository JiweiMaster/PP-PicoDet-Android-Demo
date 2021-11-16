package com.baidu.picodetncnn;

import android.app.ProgressDialog;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.DecimalFormat;


public class PhotoFragment extends Fragment {

    String video_file = "out.mp4";
//    String video_file = "test3.mp4";

    private static final String TAG = PhotoFragment.class.getSimpleName();
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    NanoDetNcnn nanoDetNcnn = new NanoDetNcnn();
    private int current_model = 0;
    private int current_cpugpu = 0;
    ImageView inferImageView;
    TextView fpsTextView;
    int count = 0;
    int frameCount = 249;
//    int frameCount = 630;
    boolean isRunThread = true;

    ProgressDialog dialog = null;
    protected Handler sender = null; // Send command to worker thread
    public static final int REQUEST_RUN_MODEL = 0;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;
    protected HandlerThread worker = null; // Worker thread to load&run model
    public Handler receiver = null; // Receive messages from worker thread


    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_photo, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        initView(view);
        start_run_model();
    }

    public void start_run_model(){
        dialog = ProgressDialog.show(getActivity(), "", "Running, please wait...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public boolean run_detect_video(){

        isRunThread = false;
        double inferTime = nanoDetNcnn.testInferTime(getActivity().getAssets(), current_model, current_cpugpu);
        double fps = frameCount*1000/inferTime;
        DecimalFormat decimalFormat = new DecimalFormat("#.00");
        String fpsStr = decimalFormat.format(fps);
        fpsTextView.setText("FPS: "+fpsStr);
        isRunThread = true;
        getShowImgThread().start();
        return true;
    }

    public void initView(@NonNull View view) {
        spinnerModel = (Spinner) view.findViewById(R.id.spinnerModel_1);
        spinnerCPUGPU = (Spinner) view.findViewById(R.id.spinnerCPUGPU_1);
        inferImageView = view.findViewById(R.id.showInferImg);
        fpsTextView = view.findViewById(R.id.fpsTextView);
        System.out.println("ncnn: copyMP42Cache");
        copyMP42Cache();

        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    isRunThread = false;
                    current_model = position;
                    System.out.println("current_model => "+current_model);
                    try{
                        isRunThread = false;
                    }catch (Exception e){
                        Log.d("ncnn: ", "stop thread ...");
                    }

                    start_run_model();
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

                    start_run_model();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        dialog.dismiss();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        dialog.dismiss();
                        Toast.makeText(getActivity(), "Run model failed!", Toast.LENGTH_SHORT).show();
                        break;
                    default:
                        break;
                }
            }
        };


        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_RUN_MODEL:
                        // Run model if model is loaded
                        if (run_detect_video()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

    }

    public Thread getShowImgThread(){
        return new Thread(new Runnable() {
            @Override
            public void run() {
                File fileDir = new File(getActivity().getExternalCacheDir()+"");
                Log.e("ncnn:","file path =>"+fileDir.getAbsolutePath());
                while(isRunThread){
                    try {
                        final String imageName = fileDir.list()[count%frameCount];
                        count ++;
                        if(imageName.contains(".jpg")){
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Bitmap bitmap = BitmapFactory.decodeFile(getActivity().getExternalCacheDir() +"/"+imageName);
                                    inferImageView.setImageBitmap(bitmap);
//                                    inferImageView.setRotation(90);
                                }
                            });
                        }
                        Thread.sleep(50);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

            }
        });

    }

    public void copyMP42Cache() {
        File file = new File(getActivity().getExternalCacheDir()+"/"+video_file);
        Log.e("ncnn: src video dir => ", file.getAbsolutePath());
        if (!file.exists()) {
            try {
                InputStream is = getActivity().getAssets().open(video_file);
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();
                FileOutputStream fos = new FileOutputStream(file);
                fos.write(buffer);
                fos.close();
            } catch (Exception e) {
                Log.e("***************ncnn: copy src file to dst file failed => ", file.getAbsolutePath());
                throw new RuntimeException(e);
            }
        } else {
            Log.e("*******************ncnn: dst file exist => ", file.getAbsolutePath());
        }
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}