# Facecam (Windows)

Put any face on your webcam, live, in Discord. Drop a photo into the window, press Start, and pick **OBS Virtual Camera** as your camera in Discord.

It's a small window around [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam). The face swap runs on your graphics card through DirectML, so NVIDIA, AMD and Intel all work.

## Install

1. Download **[Facecam-Setup.exe](https://github.com/AUstnnnnnn/facecam-windows/releases/download/setup/Facecam-Setup.exe)** and run it.
2. Windows may say "Windows protected your PC" because the installer isn't signed. Click **More info**, then **Run anyway**.
3. The installer sets up Python, the face swap engine, the face models and OBS Studio. The first install downloads about 1.5 GB, so give it a few minutes. If OBS asks for permission to install, say yes.

No admin rights needed. To update, run a newer Facecam-Setup.exe: it skips anything that's already installed, so it only takes a moment. To remove Facecam, use **Settings → Apps**. That also deletes your saved photos and presets.

## Use it

1. Open **Facecam** from the Start menu.
2. Drag a face photo into the window. You can drag it straight from Google Images in your browser, or click the box to pick a file. A clear photo of someone looking at the camera works best. Drop 3 to 5 photos of the same person (different angles and lighting) and they get blended for a closer match. **Clear** starts over with someone new.
3. Pick your webcam and press **Start**.
4. In Discord, go to **User Settings → Voice & Video → Camera** and choose **OBS Virtual Camera**.

Start Facecam before you turn your camera on in Discord. Only one app can use the webcam at a time, and Facecam has to be the one holding it.

You can drop more photos while it's running; to switch to a different person, press **Clear** first. Press **Stop** to go back to normal.

To keep a face for later, open **Presets** and choose **Save These Photos as a Preset**. Pick it from the same menu next time, even mid-call.

## If something goes wrong

- **"Camera didn't open":** another app has the webcam, usually Discord, and Facecam names it when Windows says which one. Quit that app fully (Discord keeps running in the system tray), press Start in Facecam, then set Discord's **camera** to OBS Virtual Camera. After that both can stay open.
- **"OBS Virtual Camera not found":** install OBS Studio from [obsproject.com](https://obsproject.com). Open it once, click **Start Virtual Camera**, then close OBS and press Start in Facecam again.
- **It's slow or choppy:** add some light in front of you. A darker room makes face tracking work harder. Slower PCs top out around 10 to 15 frames per second.
- **Anything else:** the details are in `facecam.log` (and `install.log` for setup problems) in `%LOCALAPPDATA%\Programs\Facecam`. Send that file over.

## Notes

Ask people before you wear their face, and don't use this to pass yourself off as someone else. The face swap model (InsightFace inswapper) is licensed for non-commercial use only.

Licensed AGPL-3.0, same as Deep-Live-Cam.
