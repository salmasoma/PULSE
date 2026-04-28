import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
import UIKit

struct HomeView: View {
    @ObservedObject var session: PULSEAppSession
    @ObservedObject var pipeline: PULSEOnDevicePipeline
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @AppStorage("pulse.local.enableReasoning") private var enableReasonedReport = true
    @State private var selectedPhotos: [PhotosPickerItem] = []
    @State private var intakeStudies: [PULSEIntakeStudy] = []
    @State private var localError: String?
    @State private var isShowingCamera = false
    @State private var isShowingFileImporter = false
    @State private var batchProgressLabel: String?

    private let uploadAnchor = "upload-workspace"

    var body: some View {
        ScrollViewReader { proxy in
            ZStack {
                PULSEAppBackground()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        heroSection {
                            withAnimation(.spring(response: 0.82, dampingFraction: 0.88)) {
                                proxy.scrollTo(uploadAnchor, anchor: .top)
                            }
                        }

                        readinessStrip

                        intakeWorkspace
                            .id(uploadAnchor)

                        if let previewImage = primaryPreviewImage {
                            previewSection(image: previewImage)
                        }

                        if intakeStudies.isEmpty == false {
                            queueSection
                        }

                        promptAndRunSection

                        if let error = localError ?? pipeline.lastError {
                            errorSection(error)
                        }

                        if let latest = session.history.first {
                            recentSection(latest)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 12)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle("Clinical Workspace")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Label("PULSE", systemImage: "cross.case.fill")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(PULSEPalette.navy)
                }
            }
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarBackground(Color.white.opacity(0.92), for: .navigationBar)
            .sheet(isPresented: $isShowingCamera) {
                CameraCaptureView { image in
                    appendStudy(image: image, source: .camera, sourceName: "Camera Capture")
                }
            }
            .fileImporter(
                isPresented: $isShowingFileImporter,
                allowedContentTypes: [.image],
                allowsMultipleSelection: true
            ) { result in
                Task {
                    await importFiles(result)
                }
            }
            .onChange(of: selectedPhotos) { _, newValue in
                Task {
                    await importPhotoLibraryItems(newValue)
                }
            }
            .navigationDestination(item: $session.presentedAnalysis) { analysis in
                ResultsView(analysis: analysis, image: session.image(for: analysis))
            }
            .navigationDestination(item: $session.presentedBatch) { batch in
                BatchResultsView(batch: batch, session: session)
            }
        }
    }

    private var primaryPreviewImage: UIImage? {
        intakeStudies.first?.image
    }

    private func heroSection(getStartedAction: @escaping () -> Void) -> some View {
        PULSEHeroSurface {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 16) {
                    VStack(alignment: .leading, spacing: 10) {
                        PULSEStatusBadge(title: "Clinical Workflow", tint: .white, systemImage: "sparkles")

                        Text("Ultrasound analysis designed for a calm clinical workflow.")
                            .font(.system(size: 34, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .fixedSize(horizontal: false, vertical: true)

                        Text("Capture or import studies, run analysis, and review a structured result with visual evidence and PDF export.")
                            .font(.subheadline)
                            .foregroundStyle(Color.white.opacity(0.82))
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer(minLength: 12)
                    PULSEBrandMark(size: 76)
                }

                Text("Designed for a private workflow today and connected reporting later.")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Color.white.opacity(0.76))

                HStack(spacing: 12) {
                    Button(action: getStartedAction) {
                        HStack(spacing: 10) {
                            Image(systemName: "arrow.down.circle.fill")
                            Text("Get Started")
                                .fontWeight(.semibold)
                        }
                        .foregroundStyle(PULSEPalette.navyDeep)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 14)
                        .background(Color.white, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                    .buttonStyle(.plain)

                    Button {
                        session.selectedTab = .history
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "clock.arrow.trianglehead.counterclockwise.rotate.90")
                            Text("History")
                                .fontWeight(.semibold)
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 14)
                        .background(Color.white.opacity(0.1), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var readinessStrip: some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "Workflow",
                subtitle: "Three steps from acquisition to structured review.",
                systemImage: "rectangle.3.group.bubble.left.fill"
            )

            Group {
                if horizontalSizeClass == .compact {
                    VStack(spacing: 12) {
                        workflowTile(number: "01", title: "Capture", detail: "Camera, gallery, or files.", icon: "camera.metering.center.weighted")
                        workflowTile(number: "02", title: "Analyze", detail: "Routing and specialist inference.", icon: "waveform.path.ecg")
                        workflowTile(number: "03", title: "Review", detail: "Open evidence, then export.", icon: "doc.richtext.fill")
                    }
                } else {
                    HStack(spacing: 12) {
                        workflowTile(number: "01", title: "Capture", detail: "Camera, gallery, or files.", icon: "camera.metering.center.weighted")
                        workflowTile(number: "02", title: "Analyze", detail: "Routing and specialist inference.", icon: "waveform.path.ecg")
                        workflowTile(number: "03", title: "Review", detail: "Open evidence, then export.", icon: "doc.richtext.fill")
                    }
                }
            }
        }
        .pulseCard(background: Color.white.opacity(0.72))
    }

    private func workflowTile(number: String, title: String, detail: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(number)
                    .font(.caption.weight(.bold))
                    .monospacedDigit()
                    .foregroundStyle(PULSEPalette.accent)
                    .lineLimit(1)
                    .minimumScaleFactor(0.9)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(PULSEPalette.accent.opacity(0.08), in: Capsule())
                Spacer()
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(PULSEPalette.accent)
                    .frame(width: 20, height: 20)
            }
            Text(title)
                .font(.headline.weight(.semibold))
                .foregroundStyle(PULSEPalette.navy)
                .lineLimit(1)
                .fixedSize(horizontal: false, vertical: true)
            Text(detail)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, minHeight: 132, alignment: .leading)
        .padding(16)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.98), Color.white.opacity(0.88)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 20, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(PULSEPalette.divider, lineWidth: 1)
        )
    }

    private var intakeWorkspace: some View {
        VStack(alignment: .leading, spacing: 16) {
            PULSESectionHeader(
                title: "Study Intake",
                subtitle: "Select one or many studies from the camera, gallery, or file system for a structured review.",
                systemImage: "cross.case.circle.fill"
            )

            VStack(spacing: 14) {
                HStack(spacing: 12) {
                    intakeSourceButton(title: "Camera", subtitle: "Capture", icon: "camera.fill") {
                        isShowingCamera = true
                    }
                    .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
                    .opacity(UIImagePickerController.isSourceTypeAvailable(.camera) ? 1 : 0.55)

                    PhotosPicker(selection: $selectedPhotos, maxSelectionCount: 0, matching: .images) {
                        intakeSourceLabel(title: "Gallery", subtitle: "Batch Select", icon: "photo.stack.fill")
                    }
                    .buttonStyle(.plain)

                    intakeSourceButton(title: "Files", subtitle: "Batch Import", icon: "folder.fill") {
                        isShowingFileImporter = true
                    }
                }

                VStack(spacing: 16) {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    PULSEPalette.accent.opacity(0.06),
                                    PULSEPalette.mint.opacity(0.08)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(height: 126)
                        .overlay {
                            VStack(spacing: 10) {
                                Image(systemName: intakeStudies.isEmpty ? "square.stack.3d.up.badge.a.fill" : "checkmark.circle.badge.plus")
                                    .font(.system(size: 32, weight: .semibold))
                                    .foregroundStyle(PULSEPalette.accent)
                                Text(intakeStudies.isEmpty ? "Start with a Study Selection" : "\(intakeStudies.count) Study\(intakeStudies.count == 1 ? "" : "ies") Selected")
                                    .font(.headline.weight(.semibold))
                                    .foregroundStyle(PULSEPalette.navy)
                                    .lineLimit(2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Text("Prepare your studies, then run a single structured review pass.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }

                    HStack(spacing: 10) {
                        intakeBullet(icon: "lock.fill", text: "Private by default")
                        intakeBullet(icon: "shippingbox.fill", text: "Distilled models")
                        intakeBullet(icon: "doc.richtext.fill", text: "PDF ready")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(18)
                .background(PULSEPalette.shellStrong, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
            }
        }
        .pulseCard()
    }

    private func intakeSourceButton(title: String, subtitle: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            intakeSourceLabel(title: title, subtitle: subtitle, icon: icon)
        }
        .buttonStyle(.plain)
    }

    private func intakeSourceLabel(title: String, subtitle: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(PULSEPalette.accent)
                .frame(width: 20, height: 20, alignment: .leading)
            Text(title)
                .font(.headline.weight(.semibold))
                .foregroundStyle(PULSEPalette.navy)
                .lineLimit(1)
                .minimumScaleFactor(0.82)
                .allowsTightening(true)
            Text(subtitle)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .allowsTightening(true)
        }
        .frame(maxWidth: .infinity, minHeight: 112, alignment: .leading)
        .padding(14)
        .background(Color.white.opacity(0.96), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(PULSEPalette.divider, lineWidth: 1)
        )
    }

    private func intakeBullet(icon: String, text: String) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.caption.weight(.bold))
                .foregroundStyle(PULSEPalette.success)
            Text(text)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
    }

    private func previewSection(image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                PULSESectionHeader(
                    title: "Study Preview",
                    subtitle: "The first selected study is shown here before analysis begins.",
                    systemImage: "viewfinder.circle.fill"
                )
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("\(Int(image.size.width)) × \(Int(image.size.height))")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(PULSEPalette.navy)
                    Text("pixels")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay(alignment: .topLeading) {
                    PULSEStatusBadge(title: "Preview", tint: PULSEPalette.accent, systemImage: "checkmark.circle.fill")
                        .padding(14)
                }
        }
        .pulseCard()
    }

    private var queueSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                PULSESectionHeader(
                    title: "Selected Studies",
                    subtitle: "Review the selected inputs before starting the structured analysis run.",
                    systemImage: "square.stack.3d.up.fill"
                )
                Spacer()
                Button("Clear All") {
                    intakeStudies.removeAll()
                }
                .font(.footnote.weight(.semibold))
                .foregroundStyle(PULSEPalette.accent)
            }

            LazyVStack(spacing: 12) {
                ForEach(Array(intakeStudies.enumerated()), id: \.element.id) { index, study in
                    intakeRow(study, index: index)
                }
            }
        }
        .pulseCard(background: Color.white.opacity(0.8))
    }

    private func intakeRow(_ study: PULSEIntakeStudy, index: Int) -> some View {
        HStack(spacing: 14) {
            Group {
                if let image = study.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    ZStack {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.white)
                        Image(systemName: study.source.symbol)
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(PULSEPalette.accent)
                    }
                }
            }
            .frame(width: 82, height: 82)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.white.opacity(0.75), lineWidth: 1)
            )

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(String(format: "%02d", index + 1))
                        .font(.caption.weight(.bold))
                        .foregroundStyle(PULSEPalette.accent)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(PULSEPalette.accent.opacity(0.08), in: Capsule())
                    PULSEInfoChip(title: study.source.title, systemImage: study.source.symbol)
                }
                Text(study.sourceName)
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                if let image = study.image {
                    Text("\(Int(image.size.width)) × \(Int(image.size.height)) pixels")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .layoutPriority(1)

            Spacer()

            Button {
                intakeStudies.removeAll { $0.id == study.id }
            } label: {
                Image(systemName: "trash.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.critical)
                    .frame(width: 36, height: 36)
                    .background(PULSEPalette.critical.opacity(0.08), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.98), Color.white.opacity(0.9)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 20, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(PULSEPalette.divider, lineWidth: 1)
        )
    }

    private var promptAndRunSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Generate reasoning report")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                Spacer()
                Toggle("", isOn: $enableReasonedReport)
                    .labelsHidden()
            }
            .padding(16)
            .background(PULSEPalette.shellStrong, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(PULSEPalette.divider, lineWidth: 1)
            )

            Button {
                Task {
                    await runAnalysis()
                }
            } label: {
                HStack(spacing: 12) {
                    if pipeline.isRunning {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Image(systemName: intakeStudies.count > 1 ? "square.stack.3d.up.fill" : "stethoscope.circle.fill")
                            .font(.system(size: 18, weight: .semibold))
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text(primaryActionTitle)
                            .font(.headline.weight(.semibold))
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(batchProgressLabel ?? primaryActionSubtitle)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.82))
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                }
                .foregroundStyle(.white)
                .padding(16)
                .frame(maxWidth: .infinity)
                .background(
                    LinearGradient(
                        colors: [PULSEPalette.navy, Color(red: 0.05, green: 0.38, blue: 0.58)],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: RoundedRectangle(cornerRadius: 22, style: .continuous)
                )
            }
            .buttonStyle(.plain)
            .disabled(intakeStudies.isEmpty || pipeline.isRunning)
            .opacity(intakeStudies.isEmpty || pipeline.isRunning ? 0.55 : 1.0)
        }
        .pulseCard()
    }

    private var primaryActionTitle: String {
        if pipeline.isRunning {
            return intakeStudies.count > 1 ? "Running Batch Analysis" : "Running Analysis"
        }
        return intakeStudies.count > 1 ? "Start Batch Analysis" : "Start Clinical Analysis"
    }

    private var primaryActionSubtitle: String {
        if intakeStudies.count > 1 {
            return "\(intakeStudies.count) selected studies will be processed sequentially and saved to the archive."
        }
        return "Routing, specialist execution, artifact generation, and structured PDF-ready reporting."
    }

    private func recentSection(_ latest: PULSESavedAnalysis) -> some View {
        Button {
            session.openHistory(latest)
        } label: {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: latest.subtitle.pulseDomainSymbol)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(PULSEPalette.accent)
                        Text("Recent Result")
                            .font(.headline.weight(.semibold))
                            .foregroundStyle(PULSEPalette.navy)
                    }

                    Text(latest.title)
                        .font(.title3.weight(.bold))
                        .foregroundStyle(PULSEPalette.navy)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(latest.displaySourceName)
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(PULSEPalette.accent)
                    Text(latest.createdAt.pulseTimestamp)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .layoutPriority(1)
                Spacer()
                Image(systemName: "arrow.right.circle.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(PULSEPalette.accent)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .pulseCard(background: Color.white.opacity(0.8))
    }

    private func errorSection(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(PULSEPalette.warning)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .pulseCard(background: Color(red: 1.0, green: 0.97, blue: 0.92))
    }

    private func importPhotoLibraryItems(_ items: [PhotosPickerItem]) async {
        guard items.isEmpty == false else { return }
        localError = nil
        let startingCount = intakeStudies.count

        for (index, item) in items.enumerated() {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data)
            else {
                continue
            }
            appendStudy(image: image, source: .photoLibrary, sourceName: "Photo Study \(startingCount + index + 1)")
        }
        selectedPhotos = []
    }

    private func importFiles(_ result: Result<[URL], Error>) async {
        localError = nil
        do {
            let urls = try result.get()
            for url in urls {
                let accessed = url.startAccessingSecurityScopedResource()
                defer {
                    if accessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }

                let data = try Data(contentsOf: url)
                guard let image = UIImage(data: data) else { continue }
                appendStudy(image: image, source: .files, sourceName: url.lastPathComponent)
            }
        } catch {
            localError = "Files could not be imported: \(error.localizedDescription)"
        }
    }

    private func appendStudy(image: UIImage, source: PULSEIntakeStudy.Source, sourceName: String) {
        guard let jpeg = image.jpegData(compressionQuality: 0.95) else {
            localError = "One study could not be prepared for local analysis."
            return
        }
        let name = sourceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Ultrasound Study" : sourceName
        intakeStudies.append(PULSEIntakeStudy(source: source, sourceName: name, imageData: jpeg))
    }

    private func runAnalysis() async {
        localError = nil
        guard intakeStudies.isEmpty == false else {
            localError = "Select at least one ultrasound study before running analysis."
            return
        }

        var savedAnalyses: [PULSESavedAnalysis] = []
        var failedStudies: [String] = []
        let shouldOpenSingleResultImmediately = intakeStudies.count == 1

        for (index, study) in intakeStudies.enumerated() {
            batchProgressLabel = "Processing \(index + 1) of \(intakeStudies.count): \(study.sourceName)"

            guard let image = study.image else {
                failedStudies.append(study.sourceName)
                continue
            }

            do {
                let baseReport = try await pipeline.analyzeReport(primaryImage: image)
                let initialReport: PULSEAnalysisReport
                if enableReasonedReport {
                    initialReport = baseReport.withReasoningStatusMessage(PULSEReasoningService.processingStatusMessage)
                } else {
                    initialReport = baseReport
                }

                let saved = try session.saveAnalysis(report: initialReport, sourceImage: image, prompt: "", sourceName: study.sourceName)
                savedAnalyses.append(saved)

                if shouldOpenSingleResultImmediately {
                    session.openHistory(saved)
                }

                if enableReasonedReport {
                    startBackgroundReasoningUpdate(
                        for: saved,
                        baseReport: baseReport,
                        image: image
                    )
                }
            } catch {
                failedStudies.append(study.sourceName)
            }
        }

        batchProgressLabel = nil

        if savedAnalyses.isEmpty {
            localError = "No studies completed successfully in this run."
            return
        }

        intakeStudies.removeAll()
        selectedPhotos = []

        if failedStudies.isEmpty == false {
            localError = "\(failedStudies.count) study\(failedStudies.count == 1 ? "" : "ies") could not be analyzed in this run."
        }

        if shouldOpenSingleResultImmediately {
            if let first = savedAnalyses.first, session.presentedAnalysis?.id != first.id {
                session.openHistory(first)
            }
        } else if savedAnalyses.count == 1, let first = savedAnalyses.first {
            session.openHistory(first)
        } else {
            session.presentBatch(savedAnalyses)
        }
    }

    private func startBackgroundReasoningUpdate(
        for saved: PULSESavedAnalysis,
        baseReport: PULSEAnalysisReport,
        image: UIImage
    ) {
        Task {
            let reasoningService = PULSEReasoningService()
            let updatedReport: PULSEAnalysisReport
            let reasoningStart = DispatchTime.now().uptimeNanoseconds
            do {
                let reasonedReport = try await reasoningService.generateReasonedReport(
                    report: baseReport,
                    primaryImage: image
                )
                let reasoningMs = Double(DispatchTime.now().uptimeNanoseconds - reasoningStart) / 1_000_000.0
                let profiling = baseReport.profiling?.withReasoningDuration(reasoningMs)
                updatedReport = baseReport
                    .withReasonedReport(reasonedReport)
                    .withProfiling(profiling)
            } catch {
                let reasoningMs = Double(DispatchTime.now().uptimeNanoseconds - reasoningStart) / 1_000_000.0
                let profiling = baseReport.profiling?.withReasoningDuration(reasoningMs)
                updatedReport = baseReport
                    .withReasoningStatusMessage(error.localizedDescription)
                    .withProfiling(profiling)
            }

            do {
                try session.updateAnalysis(saved.withReport(updatedReport))
            } catch {
                localError = "The on-device reasoning report could not be saved back to history."
            }
        }
    }
}
