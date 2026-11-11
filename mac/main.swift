import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

let bridgePath = Bundle.main.path(forResource: "bridge", ofType: "py") ?? "bridge.py"
let setupPath = Bundle.main.path(forResource: "setup", ofType: "sh") ?? "setup.sh"
// Everything setup.sh installs lives here, so removing Facecam is: delete the app and this folder.
// FACECAM_SUPPORT points it elsewhere (tests, CI); bridge.py and setup.sh read it too.
let supportDir = ProcessInfo.processInfo.environment["FACECAM_SUPPORT"].map { URL(fileURLWithPath: $0) }
    ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Facecam")
let dlcPython = supportDir.appendingPathComponent("venv/bin/python").path
let brand = Color(red: 1, green: 0x6a / 255.0, blue: 0x1a / 255.0)
let logURL = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Logs/Facecam.log")

struct Cam: Hashable { let index: Int; let name: String }

// MARK: - Engine: owns the Python sidecar and the preview session

final class Engine: ObservableObject {
    @Published var cameras: [Cam] = []
    @Published var camera = -1
    @Published var faces: [NSImage] = []   // photos of one person, blended; newest last
    var face: NSImage? { faces.last }
    @Published var running = false   // sidecar process alive
    @Published var live = false      // frames reaching the virtual camera
    @Published var swapping = false  // wearing the face; off means the camera shows you
    @Published var status = ""
    @Published var failed = false
    @Published var needsOBSSetup = false
    @Published var presets: [String] = []
    @Published var preset: String?            // name on screen until the photos change

    let session = AVCaptureSession()
    private var proc: Process?
    private var input: FileHandle?
    private var stopping = false
    private var reopen = false
    private let faceDir = supportDir.appendingPathComponent("faces")
    private let presetDir = supportDir.appendingPathComponent("presets")
    static let maxFaces = 5

    private var faceFiles: [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: faceDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    init() {
        faces = faceFiles.compactMap { NSImage(contentsOf: $0) }
        if !faces.isEmpty { status = "Ready" }
        refreshPresets()
    }

    // MARK: Presets: named copies of a photo set

    private func pngs(in dir: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func refreshPresets() {
        presets = ((try? FileManager.default.contentsOfDirectory(at: presetDir, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map(\.lastPathComponent)
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    func presetThumb(_ name: String) -> NSImage? {
        guard let first = pngs(in: presetDir.appendingPathComponent(name)).first,
              let img = NSImage(contentsOf: first) else { return nil }
        let size = NSSize(width: 18, height: 18)
        return NSImage(size: size, flipped: false) { rect in
            let side = min(img.size.width, img.size.height)
            let crop = NSRect(x: (img.size.width - side) / 2, y: (img.size.height - side) / 2, width: side, height: side)
            NSBezierPath(roundedRect: rect, xRadius: 4, yRadius: 4).addClip()
            img.draw(in: rect, from: crop, operation: .copy, fraction: 1)
            return true
        }
    }

    /// Name stripped of path separators and leading dots so it stays one folder inside presets/.
    static func presetName(_ raw: String) -> String? {
        let name = String(raw.filter { $0 != "/" && $0 != ":" && !$0.isNewline }
            .trimmingCharacters(in: .whitespaces)
            .drop(while: { $0 == "." }).prefix(40)).trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? nil : name
    }

    func askSavePreset() {
        let alert = NSAlert()
        alert.messageText = "Save Preset"
        alert.informativeText = "Name this face. You can switch to it from the Presets menu later."
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 240, height: 24))
        field.stringValue = preset ?? ""
        field.placeholderString = "Name"
        alert.accessoryView = field
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn, let name = Engine.presetName(field.stringValue) else { return }
        if presets.contains(name) {
            let confirm = NSAlert()
            confirm.messageText = "Replace \u{201C}\(name)\u{201D}?"
            confirm.informativeText = "A preset with this name already exists. Its photos will be replaced with these."
            confirm.addButton(withTitle: "Replace")
            confirm.addButton(withTitle: "Cancel")
            guard confirm.runModal() == .alertFirstButtonReturn else { return }
        }
        savePreset(name)
    }

    func savePreset(_ name: String) {
        let fm = FileManager.default, dest = presetDir.appendingPathComponent(name)
        try? fm.removeItem(at: dest)
        do {
            try fm.createDirectory(at: dest, withIntermediateDirectories: true)
            for f in faceFiles { try fm.copyItem(at: f, to: dest.appendingPathComponent(f.lastPathComponent)) }
        } catch { return fail("Couldn't save that preset. Try another name.") }
        refreshPresets()
        preset = name; failed = false; status = "Saved \u{201C}\(name)\u{201D}."
    }

    func loadPreset(_ name: String) {
        let fm = FileManager.default
        faceFiles.forEach { try? fm.removeItem(at: $0) }
        try? fm.createDirectory(at: faceDir, withIntermediateDirectories: true)
        for f in pngs(in: presetDir.appendingPathComponent(name)).suffix(Engine.maxFaces) {
            try? fm.copyItem(at: f, to: faceDir.appendingPathComponent(f.lastPathComponent))
        }
        faces = faceFiles.compactMap { NSImage(contentsOf: $0) }
        preset = name; failed = false
        if faces.isEmpty { fail("That preset has no photos.") }
        else if running { send("face \(faceDir.path)") }
        else { status = "Ready" }
    }

    func deletePreset(_ name: String) {
        try? FileManager.default.removeItem(at: presetDir.appendingPathComponent(name))
        if preset == name { preset = nil }
        refreshPresets()
    }

    func loadCameras() {
        DispatchQueue.global().async {
            let out = Pipe(), p = Process()
            p.executableURL = URL(fileURLWithPath: dlcPython)
            p.arguments = [bridgePath, "list"]
            p.standardOutput = out
            p.standardError = FileHandle.nullDevice
            do { try p.run() } catch {
                self.fail("The face engine is missing. Quit and reopen Facecam to set it up again.")
                return
            }
            let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            let cams = text.split(separator: "\n").compactMap { line -> Cam? in
                let parts = line.split(separator: "\t", maxSplits: 1)
                guard parts.count == 2, let i = Int(parts[0]) else { return nil }
                return Cam(index: i, name: String(parts[1]))
            }
            DispatchQueue.main.async {
                self.cameras = cams
                if self.camera == -1 { self.camera = cams.first?.index ?? -1 }
                if cams.isEmpty { self.fail("No camera found") } else { self.connect() }
            }
        }
    }

    // MARK: Face photo

    func take(_ providers: [NSItemProvider]) -> Bool {
        var took = false
        for p in providers {
            // Browsers offer the picture itself; Finder offers a file; a bare link gets downloaded.
            if p.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
                p.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in self.use(data) }
            } else if p.canLoadObject(ofClass: URL.self) {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return self.use(nil) }
                    if url.isFileURL { return self.use(try? Data(contentsOf: url)) }
                    URLSession.shared.dataTask(with: url) { data, _, _ in self.use(data) }.resume()
                }
            } else { continue }
            took = true
        }
        return took
    }

    func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.message = "Pick one or more photos of the same person"
        if panel.runModal() == .OK { panel.urls.forEach { use(try? Data(contentsOf: $0)) } }
    }

    func clearFaces() {
        faceFiles.forEach { try? FileManager.default.removeItem(at: $0) }
        if swapping { send("off"); swapping = false }
        faces = []; status = ""; failed = false; preset = nil
    }

    private func use(_ data: Data?) {
        // Re-encode as PNG so OpenCV never meets WebP, HEIC or a TIFF it can't read.
        guard let data, let img = NSImage(data: data), let tiff = img.tiffRepresentation,
              let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
        else { return fail("That wasn't an image. Try dragging the picture itself.") }
        DispatchQueue.main.async {
            try? FileManager.default.createDirectory(at: self.faceDir, withIntermediateDirectories: true)
            try? png.write(to: self.faceDir.appendingPathComponent("\(Int(Date().timeIntervalSince1970 * 1000)).png"), options: .atomic)
            for old in self.faceFiles.dropLast(Engine.maxFaces) { try? FileManager.default.removeItem(at: old) }
            self.faces = Array((self.faces + [img]).suffix(Engine.maxFaces))
            self.preset = nil
            self.failed = false
            if self.running { self.send("face \(self.faceDir.path)") }
            else { self.status = "Ready" }
        }
    }

    // MARK: Camera on from launch; Start/Stop scans the face in and out

    /// Opens the camera and starts the virtual camera showing you as you are.
    func connect() {
        guard !running, camera >= 0 else { return }
        AVCaptureDevice.requestAccess(for: .video) { ok in
            DispatchQueue.main.async {
                guard ok else {
                    return self.fail("Camera access is off. Turn on Facecam in System Settings › Privacy & Security › Camera.")
                }
                self.launch()
            }
        }
    }

    /// Picked a different camera: reopen on it, keeping the swap on or off as it was.
    func switchCamera() {
        guard running else { return connect() }
        reopen = true
        stopping = true
        proc?.terminate()
    }

    func toggle() {
        guard running else { return connect() }  // e.g. after setting up OBS
        swapping.toggle()
        send(swapping ? "on" : "off")
    }

    private func launch() {
        let p = Process(), out = Pipe(), inp = Pipe()
        p.executableURL = URL(fileURLWithPath: dlcPython)
        p.arguments = [bridgePath, "run", String(camera), faceDir.path]
        p.standardOutput = out
        p.standardInput = inp
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        p.standardError = (try? FileHandle(forWritingTo: logURL)) ?? FileHandle.nullDevice
        out.fileHandleForReading.readabilityHandler = { h in
            let text = String(decoding: h.availableData, as: UTF8.self)
            for line in text.split(separator: "\n") where line.hasPrefix("status ") {
                let msg = String(line.dropFirst(7))
                DispatchQueue.main.async { self.handle(msg) }
            }
        }
        p.terminationHandler = { _ in
            out.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async { self.ended() }
        }
        do { try p.run() } catch { return fail("Couldn't start the face engine") }
        proc = p; input = inp.fileHandleForWriting
        stopping = false; failed = false; needsOBSSetup = false
        running = true; status = "Starting the camera…"
        if swapping { send("on") }
    }

    private func handle(_ msg: String) {
        if msg == "Live" {
            live = true; failed = false; status = ""
            startPreview()
        } else if msg.hasPrefix("Virtual camera unavailable") {
            needsOBSSetup = true
            fail("OBS Virtual Camera isn't set up yet")
        } else if msg.hasPrefix("No face") {
            swapping = false
            fail("No face found in those photos. Try clear, front-facing ones.")
        } else if msg == "Ready" || msg == "Off" || msg == "Swapping" {
            if !failed { status = "" }
        } else {
            failed = msg.contains("didn't")
            status = msg
        }
    }

    private func ended() {
        session.stopRunning()
        if !stopping && !failed { fail("The face engine stopped. Details in ~/Library/Logs/Facecam.log") }
        running = false; live = false; proc = nil; input = nil
        if reopen { reopen = false; connect() } else { swapping = false }
    }

    private func send(_ line: String) { try? input?.write(contentsOf: Data((line + "\n").utf8)) }

    private func fail(_ msg: String) {
        let apply = { self.status = msg; self.failed = true }
        Thread.isMainThread ? apply() : DispatchQueue.main.async(execute: apply)
    }

    func setUpOBS() {
        // OBS installs and starts its camera extension; macOS then asks to allow it once.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        p.arguments = ["-a", "OBS", "--args", "--startvirtualcam"]
        try? p.run()
        status = "Allow the OBS camera extension if macOS asks, then quit OBS and press Start."
        failed = false; needsOBSSetup = false
    }

    private func startPreview() {
        let devices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external], mediaType: .video, position: .unspecified).devices
        guard let dev = devices.first(where: { $0.localizedName.contains("OBS") }),
              let inp = try? AVCaptureDeviceInput(device: dev) else { return }
        session.beginConfiguration()
        session.inputs.forEach(session.removeInput)
        if session.canAddInput(inp) { session.addInput(inp) }
        session.commitConfiguration()
        DispatchQueue.global().async { self.session.startRunning() }
    }

    func quit() { proc?.terminate() }
}

// MARK: - Views

struct PreviewView: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        v.layer = layer
        v.wantsLayer = true
        return v
    }
    func updateNSView(_ v: NSView, context: Context) {}
}

var snapshotting = false

// ImageRenderer can't draw AppKit drop targets, so snapshots skip it.
struct DropTarget: ViewModifier {
    let engine: Engine
    @Binding var targeted: Bool
    func body(content: Content) -> some View {
        if snapshotting { content }
        else { content.onDrop(of: [.image, .fileURL, .url], isTargeted: $targeted, perform: engine.take) }
    }
}

/// Pop-up look with the chevron badge in Facecam orange. The system pop-ups always
/// use the Mac's own accent colour, whatever tint the app asks for.
struct DropdownLabel: View {
    let text: String
    let symbol: String
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        HStack(spacing: 8) {
            Text(text).lineLimit(1).truncationMode(.tail)
            Spacer(minLength: 0)
            Image(systemName: symbol)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 18, height: 18)
                .background(enabled ? brand : Color.secondary.opacity(0.5), in: RoundedRectangle(cornerRadius: 5, style: .continuous))
        }
        .padding(.leading, 9).padding(.trailing, 3).frame(height: 24)
        .background(Color(nsColor: .controlColor), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Color.primary.opacity(0.12)))
        .shadow(color: .black.opacity(0.12), radius: 0.5, y: 0.5)
        .contentShape(Rectangle())
    }
}

struct ContentView: View {
    @ObservedObject var engine: Engine
    @State private var targeted = false
    let corner = RoundedRectangle(cornerRadius: 12, style: .continuous)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            stage
            HStack(spacing: 8) {
                presetsMenu
                Menu {
                    ForEach(engine.cameras, id: \.index) { cam in
                        Button { engine.camera = cam.index } label: {
                            if cam.index == engine.camera { Label(cam.name, systemImage: "checkmark") } else { Text(cam.name) }
                        }
                    }
                } label: {
                    DropdownLabel(text: engine.cameras.first { $0.index == engine.camera }?.name ?? "Finding cameras…",
                                  symbol: "chevron.up.chevron.down")
                }
                .menuStyle(.button).buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(engine.cameras.isEmpty)
                .onChange(of: engine.camera) { old, _ in if old >= 0 { engine.switchCamera() } }
                Button(action: engine.toggle) {
                    Text(engine.swapping ? "Stop" : "Start").frame(minWidth: 56)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .tint(engine.swapping ? .red : brand)
                .keyboardShortcut(.defaultAction)
                .disabled(!engine.swapping && (engine.face == nil || engine.camera < 0 || (engine.running && !engine.live)))
                .help(engine.swapping ? "Scan back to your own face" : "Scan their face onto yours")
            }
            statusLine
        }
        .padding(16)
        .frame(width: 400)
    }

    var presetsMenu: some View {
        Menu {
            ForEach(engine.presets, id: \.self) { name in
                Button { engine.loadPreset(name) } label: {
                    if let thumb = engine.presetThumb(name) {
                        Label { Text(name) } icon: { Image(nsImage: thumb) }
                    } else { Text(name) }
                }
            }
            if engine.presets.isEmpty { Text("No presets yet") }
            Divider()
            Button("Save These Photos as a Preset…", action: engine.askSavePreset)
                .disabled(engine.faces.isEmpty)
            if !engine.presets.isEmpty {
                Menu("Delete") {
                    ForEach(engine.presets, id: \.self) { name in Button(name) { engine.deletePreset(name) } }
                }
            }
        } label: {
            DropdownLabel(text: engine.preset ?? "Presets", symbol: "chevron.down")
        }
        .menuStyle(.button).buttonStyle(.plain)
        .fixedSize()
        .help("Save these photos as a preset, or switch to a saved one")
    }

    var stage: some View {
        corner.fill(.black.opacity(engine.face == nil && !engine.live ? 0.06 : 1))
            .aspectRatio(16 / 9, contentMode: .fit)
            .overlay { stageContent }
            .clipShape(corner)
            .overlay(
                corner.strokeBorder(targeted ? brand : Color.secondary.opacity(engine.face == nil ? 0.5 : 0.15),
                                    style: StrokeStyle(lineWidth: targeted ? 2 : 1, dash: engine.face == nil && !engine.live && !targeted ? [5, 4] : []))
            )
            .contentShape(corner)
            .onTapGesture { if !engine.live || engine.faces.isEmpty { engine.choose() } }
            .modifier(DropTarget(engine: engine, targeted: $targeted))
            .animation(.easeOut(duration: 0.15), value: targeted)
    }

    var stageContent: some View {
        ZStack {
            if engine.live {
                PreviewView(session: engine.session)
                if engine.faces.isEmpty && !targeted {
                    Label("Drop a face photo to wear it", systemImage: "face.dashed")
                        .font(.callout.weight(.medium)).foregroundStyle(.white)
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(.black.opacity(0.55), in: Capsule())
                        .padding(12)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                }
            } else if let face = engine.face {
                // Fill inside a fixed box so the photo can't stretch the stack past the stage.
                Color.clear.overlay { Image(nsImage: face).resizable().scaledToFill() }.clipped()
                    .opacity(engine.running ? 0.45 : 1)
                if engine.running { ProgressView().controlSize(.small).tint(.white) }
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "face.dashed").font(.system(size: 30, weight: .light))
                        .foregroundStyle(.secondary)
                    Text("Drop a face photo").font(.headline)
                    Text("Drag one from your browser, or click to choose.\nA few photos of the same person match closer.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
            }
            if engine.faces.count > (engine.live ? 0 : 1) {
                HStack(spacing: 6) {
                    ForEach(Array(engine.faces.enumerated()), id: \.offset) { _, img in
                        Image(nsImage: img).resizable().scaledToFill()
                            .frame(width: 36, height: 36)
                            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous)
                                .strokeBorder(.white.opacity(0.9), lineWidth: 1.5))
                    }
                }
                .shadow(color: .black.opacity(0.35), radius: 4, y: 2)
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                .help("These photos are blended into one face")
            }
            if !engine.faces.isEmpty && !targeted {
                Button(action: engine.clearFaces) {
                    Label("Clear", systemImage: "xmark").font(.caption.weight(.semibold))
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.black.opacity(0.55), in: Capsule())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .help("Remove all photos to start on a different person")
            }
            if targeted {
                corner.fill(.black.opacity(0.35))
                Text(engine.faces.isEmpty ? "Drop to use this face" : "Drop to add this photo")
                    .font(.headline).foregroundStyle(.white)
            }
        }
    }

    var statusLine: some View {
        HStack(spacing: 6) {
            if engine.failed || (!engine.status.isEmpty && !engine.live && engine.status != "Ready") {
                Text(engine.status).foregroundStyle(engine.failed ? Color.red : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !engine.live && engine.status == "Ready" {
                Text(engine.faces.count == 1 ? "Ready. Add more photos of them for a closer match."
                                             : "Ready. Blending \(engine.faces.count) photos.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else if engine.live && engine.swapping {
                Circle().fill(.red).frame(width: 7, height: 7)
                Text("Live").fontWeight(.semibold)
                Text("In Discord, pick OBS Virtual Camera").foregroundStyle(.secondary)
            } else if engine.live {
                Circle().fill(.secondary).frame(width: 7, height: 7)
                Text(engine.face == nil ? "Camera on, showing you. Drop a photo to wear their face."
                     : engine.faces.count == 1 ? "Showing you. Press Start to switch faces."
                     : "Showing you. Press Start to wear a blend of \(engine.faces.count) photos.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else if !engine.status.isEmpty {
                Text(engine.status).foregroundStyle(.secondary)
            } else {
                Text("Pick a face, then press Start").foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if engine.needsOBSSetup { Button("Set Up", action: engine.setUpOBS).controlSize(.small) }
        }
        .font(.callout)
        .frame(minHeight: 18)
    }
}

// MARK: - First run: install the engine, allow the camera, turn on OBS's virtual camera

let setupLogURL = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Logs/Facecam-setup.log")

final class Setup: ObservableObject {
    @Published var installed = FileManager.default.fileExists(atPath: supportDir.appendingPathComponent("ready").path)
    @Published var installing = false
    @Published var step = 0                  // setup.sh's "[n/5]" counter
    @Published var stepName = ""
    @Published var installFailed = false
    @Published var camera = AVCaptureDevice.authorizationStatus(for: .video)
    @Published var vcam = Setup.hasVirtualCamera()
    @Published var obsOpened = false
    static let steps = 5
    var done: Bool { installed && camera == .authorized && vcam }
    private var poll: Timer?

    static func hasVirtualCamera() -> Bool {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.external], mediaType: .video, position: .unspecified)
            .devices.contains { $0.localizedName.contains("OBS") }
    }

    func install() {
        let p = Process(), out = Pipe()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = [setupPath]
        p.standardOutput = out; p.standardError = out
        try? FileManager.default.createDirectory(at: setupLogURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: setupLogURL.path, contents: nil)
        let log = try? FileHandle(forWritingTo: setupLogURL)
        out.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            log?.write(data)
            for line in String(decoding: data, as: UTF8.self).split(separator: "\n") where line.hasPrefix("[") {
                // "[3/5] Python and packages"
                guard let close = line.firstIndex(of: "]"), let n = Int(line.dropFirst().prefix(while: { $0 != "/" })) else { continue }
                let name = line[line.index(after: close)...].trimmingCharacters(in: .whitespaces)
                DispatchQueue.main.async { self.step = n; self.stepName = name }
            }
        }
        p.terminationHandler = { p in
            out.fileHandleForReading.readabilityHandler = nil
            try? log?.close()
            DispatchQueue.main.async {
                self.installing = false
                self.installed = p.terminationStatus == 0
                self.installFailed = !self.installed
            }
        }
        installing = true; installFailed = false; step = 0; stepName = "Starting"
        do { try p.run() } catch { installing = false; installFailed = true }
    }

    func allowCamera() {
        if camera == .notDetermined {
            AVCaptureDevice.requestAccess(for: .video) { _ in
                DispatchQueue.main.async { self.camera = AVCaptureDevice.authorizationStatus(for: .video) }
            }
        } else {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera")!)
            watch()
        }
    }

    func turnOnVirtualCamera() {
        // OBS installs its camera extension on first start; macOS then asks to allow it once.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        p.arguments = ["-a", "OBS", "--args", "--startvirtualcam"]
        try? p.run()
        obsOpened = true
        watch()
    }

    func openExtensionSettings() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.LoginItems-Settings.extension")!)
    }

    func showLog() { NSWorkspace.shared.activateFileViewerSelecting([setupLogURL]) }

    /// Both approvals happen in System Settings, so check back until they land.
    private func watch() {
        poll?.invalidate()
        poll = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] t in
            guard let self else { return t.invalidate() }
            self.camera = AVCaptureDevice.authorizationStatus(for: .video)
            self.vcam = Setup.hasVirtualCamera()
            if self.camera == .authorized && self.vcam { t.invalidate() }
        }
    }
}

struct SetupView: View {
    @ObservedObject var setup: Setup
    let finish: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 52, height: 52)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Set up Facecam").font(.title2.weight(.semibold))
                    Text("Three steps. You only do this once.").foregroundStyle(.secondary)
                }
            }
            .padding(.bottom, 18)

            row(1, "Install the face engine", done: setup.installed, active: true) { engineDetail }
            Divider().padding(.leading, 38)
            row(2, "Allow the camera", done: setup.camera == .authorized, active: setup.installed) { cameraDetail }
            Divider().padding(.leading, 38)
            row(3, "Turn on the virtual camera", done: setup.vcam,
                active: setup.installed && setup.camera == .authorized) { vcamDetail }

            HStack {
                Text(setup.done ? "All set. In Discord, pick OBS Virtual Camera as your camera."
                                : "Facecam needs all three before it can start.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 12)
                Button("Continue", action: finish)
                    .controlSize(.large).buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!setup.done)
            }
            .padding(.top, 18)
        }
        .padding(20)
        .frame(width: 440)
        .tint(brand)
    }

    func row<D: View>(_ n: Int, _ title: String, done: Bool, active: Bool, @ViewBuilder detail: () -> D) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(done ? brand : Color.secondary.opacity(active ? 0.18 : 0.08))
                if done { Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(.white) }
                else { Text("\(n)").font(.callout.weight(.semibold).monospacedDigit()).foregroundStyle(active ? .primary : .tertiary) }
            }
            .frame(width: 26, height: 26)
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.headline).foregroundStyle(active || done ? .primary : .tertiary)
                if active && !done { detail() }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 12)
    }

    @ViewBuilder var engineDetail: some View {
        if setup.installing {
            Text("\(setup.stepName)…").font(.callout).foregroundStyle(.secondary)
            // Drawn by hand: ProgressView doesn't render in --snapshot.
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.18))
                    Capsule().fill(brand).frame(width: max(6, g.size.width * CGFloat(setup.step) / CGFloat(Setup.steps)))
                }
            }
            .frame(height: 5)
            .animation(.easeOut(duration: 0.4), value: setup.step)
            Text("Downloads about 1.5 GB, so this can take a while.").font(.caption).foregroundStyle(.secondary)
        } else if setup.installFailed {
            Text("Setup stopped at \u{201C}\(setup.stepName)\u{201D}. Check your internet connection and try again; it picks up where it left off.")
                .font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            HStack { Button("Try Again", action: setup.install); Button("Show Log", action: setup.showLog) }
        } else {
            Text("Downloads Python, the face swap engine, its models and OBS. About 1.5 GB.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button("Install", action: setup.install).buttonStyle(.borderedProminent)
        }
    }

    @ViewBuilder var cameraDetail: some View {
        if setup.camera == .notDetermined {
            Text("Facecam reads your webcam to swap your face. It never leaves this Mac.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button("Allow Camera", action: setup.allowCamera).buttonStyle(.borderedProminent)
        } else {
            Text("Camera access is off. In System Settings, switch on Facecam under Camera.")
                .font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            Button("Open System Settings", action: setup.allowCamera)
        }
    }

    @ViewBuilder var vcamDetail: some View {
        if !setup.obsOpened {
            Text("OBS adds a camera called OBS Virtual Camera for Discord to use. macOS asks you to allow it once.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button("Turn On", action: setup.turnOnVirtualCamera).buttonStyle(.borderedProminent)
        } else {
            Text("When macOS says a system extension was blocked, open System Settings and switch on OBS under Camera Extensions. Then quit OBS.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack { Button("Open System Settings", action: setup.openExtensionSettings); Button("Open OBS Again", action: setup.turnOnVirtualCamera) }
        }
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    let engine = Engine()
    var window: NSWindow!

    func applicationDidFinishLaunching(_ n: Notification) {
        window = NSWindow(contentRect: .zero, styleMask: [.titled, .closable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Facecam"
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        let setup = Setup()
        if setup.done { showMain() } else {
            show(SetupView(setup: setup, finish: { [weak self] in self?.showMain() }))
        }
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func show<V: View>(_ view: V) {
        window.contentView = NSHostingView(rootView: view)
        window.setContentSize(window.contentView!.fittingSize)
    }

    func showMain() {
        window.level = .floating  // stays over Discord while you call
        show(ContentView(engine: engine))
        engine.loadCameras()
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ a: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { engine.quit() }
}

/// Draws controls as they look in the focused window, for snapshots.
final class KeyWindow: NSWindow { override var isKeyWindow: Bool { true }; override var isMainWindow: Bool { true } }

// Offscreen render of each state: Facecam --snapshot <dir> [face.png]
@MainActor func snapshot(_ dir: String, _ facePath: String?) {
    NSApp.setActivationPolicy(.prohibited)
    snapshotting = true
    let face = facePath.flatMap { NSImage(contentsOfFile: $0) }
    let one = face.map { [$0] } ?? [], three = Array(repeating: one, count: 3).flatMap { $0 }
    let cams = [Cam(index: 1200, name: "MacBook Pro Camera")]
    let states: [(String, (Engine) -> Void)] = [
        ("empty", { $0.faces = []; $0.status = "" }),
        ("ready", { $0.faces = one; $0.status = "Ready" }),
        ("blend", { $0.faces = three; $0.status = "Ready" }),
        ("preset", { $0.faces = three; $0.preset = "Gorilla"; $0.status = "Ready" }),
        ("starting", { $0.faces = one; $0.running = true; $0.status = "Starting the camera…" }),
        ("you", { $0.faces = three; $0.running = true; $0.live = true }),
        ("you-noface", { $0.faces = []; $0.running = true; $0.live = true }),
        ("live", { $0.faces = three; $0.running = true; $0.live = true; $0.swapping = true }),
        ("obs", { $0.faces = one; $0.failed = true; $0.needsOBSSetup = true; $0.status = "OBS Virtual Camera isn't set up yet" }),
    ]
    let setupStates: [(String, (Setup) -> Void)] = [
        ("setup-start", { $0.installed = false; $0.camera = .notDetermined; $0.vcam = false }),
        ("setup-installing", { $0.installed = false; $0.installing = true; $0.step = 3; $0.stepName = "Python and packages"; $0.camera = .notDetermined; $0.vcam = false }),
        ("setup-failed", { $0.installed = false; $0.installFailed = true; $0.step = 4; $0.stepName = "Face models (about 800 MB)"; $0.camera = .notDetermined; $0.vcam = false }),
        ("setup-camera", { $0.installed = true; $0.camera = .notDetermined; $0.vcam = false }),
        ("setup-denied", { $0.installed = true; $0.camera = .denied; $0.vcam = false }),
        ("setup-obs", { $0.installed = true; $0.camera = .authorized; $0.vcam = false; $0.obsOpened = true }),
        ("setup-done", { $0.installed = true; $0.camera = .authorized; $0.vcam = true }),
    ]
    for scheme in [ColorScheme.light, .dark] {
        NSApp.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
        let views: [(String, AnyView)] = states.map { name, setup in
            let e = Engine(); e.cameras = cams; e.camera = 1200; setup(e)
            return (name, AnyView(ContentView(engine: e)))
        } + setupStates.map { name, apply in
            let s = Setup(); apply(s)
            return (name, AnyView(SetupView(setup: s, finish: {})))
        }
        for (name, content) in views {
            // A borderless window parked off every screen, so AppKit controls (buttons, menus)
            // draw too; ImageRenderer leaves them as placeholders. Never ordered front.
            let host = NSHostingView(rootView: content
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, scheme)
                .environment(\.controlActiveState, .key))
            let w = KeyWindow(contentRect: NSRect(x: -10000, y: -10000, width: 10, height: 10),
                             styleMask: .borderless, backing: .buffered, defer: false)
            w.appearance = NSApp.appearance
            w.contentView = host
            w.setContentSize(host.fittingSize)
            host.layoutSubtreeIfNeeded()
            RunLoop.main.run(until: Date().addingTimeInterval(0.05))
            if let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
                host.cacheDisplay(in: host.bounds, to: rep)
                try? rep.representation(using: .png, properties: [:])?
                    .write(to: URL(fileURLWithPath: "\(dir)/\(name)-\(scheme == .dark ? "dark" : "light").png"))
            }
        }
    }
}

// Facecam --presettest <face.png>, run with FACECAM_SUPPORT set to an empty folder.
@MainActor func presetTest(_ facePath: String) {
    func check(_ ok: Bool, _ what: String) { if !ok { print("FAIL", what); exit(1) } }
    check(ProcessInfo.processInfo.environment["FACECAM_SUPPORT"] != nil, "FACECAM_SUPPORT not set")
    let data = try! Data(contentsOf: URL(fileURLWithPath: facePath))
    let e = Engine()
    for _ in 0..<3 { e.take([NSItemProvider(item: data as NSData, typeIdentifier: UTType.png.identifier)]) == true ? () : check(false, "take") }
    let until = Date().addingTimeInterval(5)
    while e.faces.count < 3 && Date() < until { RunLoop.main.run(until: Date().addingTimeInterval(0.05)) }
    check(e.faces.count == 3, "3 photos added")
    e.savePreset("Ludwig")
    check(e.presets == ["Ludwig"] && e.preset == "Ludwig", "save")
    check(Engine.presetName("../../evil:/x") == "evilx", "path tricks collapse to one folder name")
    check(Engine.presetName(" ... ") == nil, "empty name")
    e.clearFaces()
    check(e.faces.isEmpty && e.preset == nil, "clear")
    e.loadPreset("Ludwig")
    check(e.faces.count == 3 && e.preset == "Ludwig" && e.status == "Ready", "load")
    check(e.presetThumb("Ludwig") != nil, "thumb")
    e.deletePreset("Ludwig")
    check(e.presets.isEmpty && e.preset == nil && e.faces.count == 3, "delete keeps current photos")
    print("presets ok")
}

let args = CommandLine.arguments
if let i = args.firstIndex(of: "--presettest"), i + 1 < args.count {
    _ = NSApplication.shared
    MainActor.assumeIsolated { presetTest(args[i + 1]) }
    exit(0)
}
if let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count {
    _ = NSApplication.shared
    MainActor.assumeIsolated { snapshot(args[i + 1], i + 2 < args.count ? args[i + 2] : nil) }
    exit(0)
}
let delegate = AppDelegate()
NSApplication.shared.delegate = delegate
NSApplication.shared.setActivationPolicy(.regular)
NSApplication.shared.run()
