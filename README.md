# Facecam (Windows)

Put any face on your webcam, live, in Discord. Drop a photo into the window, press Start, and pick **OBS Virtual Camera** as your camera in Discord.

It's a small window around [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam). The face swap runs on your graphics card through DirectML, so NVIDIA, AMD and Intel all work.

## Setup (once)

1. Click the green **Code** button above, then **Download ZIP**, and unzip it somewhere like your Documents folder.
2. Double-click **install.bat**. It downloads Python, the face swap engine and OBS Studio. That takes 5 to 10 minutes. If Windows SmartScreen warns you, click **More info**, then **Run anyway**.

## Use it

1. Double-click **Facecam.bat**.
2. Drag a face photo into the window. You can drag it straight from Google Images in your browser, or click the box to pick a file. A clear photo of someone looking at the camera works best.
3. Pick your webcam and press **Start**. The first start downloads the face models (about 500 MB), so give it a minute.
4. In Discord, go to **User Settings → Voice & Video → Camera** and choose **OBS Virtual Camera**.

You can drop a different photo while it's running to switch faces. Press **Stop** to go back to normal.

## If something goes wrong

- **"OBS Virtual Camera not found":** install OBS Studio from [obsproject.com](https://obsproject.com). Open it once, click **Start Virtual Camera**, then close OBS and press Start in Facecam again.
- **"Camera didn't open":** close any other app using the webcam (Discord's camera preview counts), then press Start again.
- **It's slow or choppy:** move closer to the light. A darker room makes face detection work harder. Slower PCs top out around 10 to 15 frames per second.

## Notes

Ask people before you wear their face, and don't use this to pass yourself off as someone else. The face swap model (InsightFace inswapper) is licensed for non-commercial use only.

Licensed AGPL-3.0, same as Deep-Live-Cam.
