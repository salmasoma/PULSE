import SwiftUI
import UIKit

private struct PULSEVQAMessage: Identifiable, Hashable {
    enum Role {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    let text: String

    init(id: UUID = UUID(), role: Role, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }
}

@MainActor
private enum PULSEVQAConversationStore {
    private static var conversations: [UUID: [PULSEVQAMessage]] = [:]

    static func messages(for analysisID: UUID) -> [PULSEVQAMessage]? {
        conversations[analysisID]
    }

    static func save(_ messages: [PULSEVQAMessage], for analysisID: UUID) {
        conversations[analysisID] = messages
    }
}

struct PULSEVQAChatView: View {
    let analysis: PULSESavedAnalysis
    let image: UIImage

    @State private var messages: [PULSEVQAMessage] = []
    @State private var question: String = ""
    @State private var isGenerating = false
    @State private var localError: String?
    @FocusState private var isInputFocused: Bool

    var body: some View {
        ZStack {
            PULSEAppBackground()

            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            headerCard

                            ForEach(messages) { message in
                                messageBubble(message)
                            }

                            if isGenerating {
                                processingBubble
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.top, 12)
                        .padding(.bottom, 20)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: messages.count) { _, _ in
                        scrollToBottom(proxy: proxy)
                    }
                    .onChange(of: isGenerating) { _, _ in
                        scrollToBottom(proxy: proxy)
                    }
                    .onTapGesture {
                        isInputFocused = false
                    }
                }

                composerBar
            }
        }
        .navigationTitle("Ask About Image")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: analysis.id) {
            loadConversation()
            if PULSEMoondream2LocalExecutor.isRuntimeReady {
                await PULSEMoondream2LocalExecutor.prewarm()
            }
        }
        .alert("Question Failed", isPresented: Binding(
            get: { localError != nil },
            set: { if !$0 { localError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(localError ?? "The question could not be answered.")
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "Image-Grounded Q&A",
                subtitle: "Ask follow-up questions about this ultrasound frame. Answers are generated locally from the image on device.",
                systemImage: "bubble.left.and.text.bubble.right.fill"
            )

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

            HStack(spacing: 10) {
                PULSEInfoChip(title: analysis.report.detectedDomain.pulseDisplayText, systemImage: analysis.report.detectedDomain.pulseDomainSymbol)
                PULSEInfoChip(title: (analysis.report.subviewText ?? analysis.report.detectedLabel).pulseDisplayText, systemImage: "scope", tint: PULSEPalette.teal)
                if let confidence = analysis.report.primaryConfidencePercentText {
                    PULSEInfoChip(title: confidence, systemImage: "percent", tint: PULSEPalette.accent)
                }
            }
        }
        .pulseCard(background: Color.white.opacity(0.82))
    }

    private var composerBar: some View {
        VStack(spacing: 10) {
            if isGenerating {
                HStack(spacing: 10) {
                    ProgressView()
                        .tint(PULSEPalette.accent)
                    Text("Generating answer locally...")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 20)
            }

            HStack(alignment: .bottom, spacing: 12) {
                TextField("Ask a question about this image…", text: $question, axis: .vertical)
                    .textFieldStyle(.plain)
                    .focused($isInputFocused)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .background(Color.white, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .lineLimit(1...5)
                    .submitLabel(.send)
                    .onSubmit {
                        if canSend {
                            Task {
                                await askQuestion()
                            }
                        }
                    }

                Button {
                    Task {
                        await askQuestion()
                    }
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 32, weight: .semibold))
                        .foregroundStyle(canSend ? PULSEPalette.accent : PULSEPalette.divider)
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 18)
            .background(.ultraThinMaterial)
        }
    }

    private var processingBubble: some View {
        HStack {
            HStack(spacing: 10) {
                ProgressView()
                    .tint(PULSEPalette.accent)
                Text("Thinking…")
                    .font(.subheadline)
                    .foregroundStyle(PULSEPalette.navy)
            }
            .padding(14)
            .background(Color.white, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

            Spacer()
        }
        .id("processing")
    }

    private var canSend: Bool {
        question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false && !isGenerating
    }

    private func messageBubble(_ message: PULSEVQAMessage) -> some View {
        HStack {
            if message.role == .assistant {
                bubbleBody(message, background: Color.white, foreground: PULSEPalette.navy)
                Spacer(minLength: 36)
            } else {
                Spacer(minLength: 36)
                bubbleBody(message, background: PULSEPalette.accent, foreground: .white)
            }
        }
        .id(message.id)
    }

    private func bubbleBody(_ message: PULSEVQAMessage, background: Color, foreground: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(message.role == .assistant ? "VQA" : "You")
                .font(.caption.weight(.bold))
                .foregroundStyle(message.role == .assistant ? PULSEPalette.accent : Color.white.opacity(0.82))
            Text(message.text)
                .font(.subheadline)
                .foregroundStyle(foreground)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .background(background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(message.role == .assistant ? PULSEPalette.divider : Color.clear, lineWidth: 1)
        )
    }

    private func loadConversation() {
        if let storedMessages = PULSEVQAConversationStore.messages(for: analysis.id), storedMessages.isEmpty == false {
            messages = storedMessages
            return
        }

        messages = []
        PULSEVQAConversationStore.save(messages, for: analysis.id)
    }

    @MainActor
    private func askQuestion() async {
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedQuestion.isEmpty == false else { return }

        isInputFocused = false
        messages.append(PULSEVQAMessage(role: .user, text: trimmedQuestion))
        PULSEVQAConversationStore.save(messages, for: analysis.id)
        question = ""
        isGenerating = true
        defer { isGenerating = false }

        do {
            let answer = try await PULSEMoondream2LocalExecutor.answerQuestion(
                report: analysis.report,
                image: image,
                question: trimmedQuestion
            )
            messages.append(PULSEVQAMessage(role: .assistant, text: answer))
            PULSEVQAConversationStore.save(messages, for: analysis.id)
        } catch {
            messages.append(
                PULSEVQAMessage(
                    role: .assistant,
                    text: "The on-device VQA model could not answer this question reliably for the current image."
                )
            )
            PULSEVQAConversationStore.save(messages, for: analysis.id)
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        if let last = messages.last {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        } else if isGenerating {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo("processing", anchor: .bottom)
            }
        }
    }
}

private enum PULSEStructuredVQAResponder {
    enum AnswerMode {
        case structured
        case imageModel
    }

    static func preferredMode(question: String, report: PULSEAnalysisReport) -> AnswerMode {
        let q = question.lowercased()
        if asksForPureVisualDescription(q) || asksForImageOnlyDescription(q) {
            return .imageModel
        }
        return .structured
    }

    static func fallbackAnswer(question: String, report: PULSEAnalysisReport, imageSize: CGSize) -> String? {
        let q = question.lowercased()
        if asksForPureVisualDescription(q) || asksForImageOnlyDescription(q) {
            return "The image-only model could not provide a reliable visual description for this frame. Based on the structured on-device analysis, \(PULSEReasoningService.structuredFindingsSummary(for: report, imageSize: imageSize))"
        }
        return answer(question: question, report: report, imageSize: imageSize)
    }

    static func shouldAnswerDirectly(question: String, report: PULSEAnalysisReport) -> Bool {
        let q = question.lowercased()
        return asksAboutOrganOrView(q)
            || asksAboutMeasurement(q)
            || asksAboutSegmentationOrSize(q)
            || asksAboutQualityOrConfidence(q)
            || asksAboutNormality(q)
            || asksAboutPresenceAbsence(q)
            || asksForImpression(q)
            || asksWhyOrReasoning(q)
            || asksForClassSelection(q)
    }

    static func answer(question: String, report: PULSEAnalysisReport, imageSize: CGSize) -> String {
        let q = question.lowercased()

        if asksWhyOrReasoning(q) || asksForClassSelection(q) {
            return reasoningAnswer(question: q, report: report, imageSize: imageSize)
        }

        if asksAboutOrganOrView(q) {
            let primary = (report.subviewText ?? report.detectedLabel).pulseDisplayText
            var parts = ["Based on the on-device analysis, this frame is assigned to \(primary) within the \(report.detectedDomain.pulseDisplayText) domain."]
            if let rationale = PULSEReasoningService.structuredSelectionRationale(for: report) {
                parts.append(rationale)
            }
            return parts.joined(separator: " ")
        }

        if asksAboutPresenceAbsence(q) {
            if let presenceAnswer = presenceAbsenceAnswer(for: q, report: report) {
                return presenceAnswer
            }
            let primary = (report.subviewText ?? report.detectedLabel).pulseDisplayText
            return "The available structured outputs do not reliably confirm the presence or absence of that specific structure on this single still image. Current outputs primarily support \(primary) within the \(report.detectedDomain.pulseDisplayText) domain."
        }

        if asksAboutMeasurement(q), let measurement = report.specialistFindings.first(where: { $0.category == .measurement }) {
            let label = measurement.label?.pulseDisplayText.uppercased() ?? "measurement"
            let details = measurement.displayDetails.prefix(3).joined(separator: "; ")
            if details.isEmpty == false {
                return "The on-device analysis generated a \(label) measurement on this image: \(details)."
            }
            return "The on-device analysis generated a \(label) measurement on this image."
        }

        if asksAboutSegmentationOrSize(q), let segmentation = report.specialistFindings.first(where: { $0.category == .segmentation }) {
            let details = segmentation.displayDetails.prefix(5).joined(separator: "; ")
            if details.isEmpty == false {
                return "Segmentation output is available for this image with the following extent details: \(details)."
            }
            return "Segmentation output is available for this image, but no additional extent details were recorded."
        }

        if asksAboutQualityOrConfidence(q) {
            var parts: [String] = []
            parts.append("Image quality is \(report.quality.qualityLabel.capitalized).")
            if let confidence = report.primaryConfidencePercentText {
                parts.append("Primary structured confidence is \(confidence).")
            }
            return parts.joined(separator: " ")
        }

        if asksAboutNormality(q) {
            if let pathology = primaryPathologyFinding(in: report), let label = pathology.label?.pulseDisplayText {
                let normalizedLabel = label.lowercased()
                var answer = directBinaryLead(question: q, positive: normalizedLabelDoesSupportPositive(question: q, label: normalizedLabel))
                answer += "The strongest structured assessment favors \(normalizedLabel)"
                if let confidence = pathology.confidencePercentText {
                    answer += " at \(confidence)"
                }
                answer += "."
                if let rationale = PULSEReasoningService.structuredSelectionRationale(for: report) {
                    answer += " \(rationale)"
                }
                return answer
            }
            return "\(directBinaryLead(question: q, positive: nil))A definitive normal-versus-abnormal answer is not reliable from this single still image alone. The available outputs primarily support \(report.detectedDomain.pulseDisplayText.lowercased()) view or anatomy identification."
        }

        if asksForImpression(q) {
            return report.reasonedReport?.output.components(separatedBy: "\n\nIMPRESSION\n").last?.components(separatedBy: "\n\nLIMITATIONS").first
                ?? report.impressionText
        }

        return PULSEReasoningService.structuredFindingsSummary(for: report, imageSize: imageSize)
    }

    private static func asksAboutOrganOrView(_ q: String) -> Bool {
        q.contains("organ") || q.contains("view") || q.contains("what is shown") || q.contains("what do you see")
    }

    private static func asksAboutMeasurement(_ q: String) -> Bool {
        q.contains("measurement") || q.contains("length") || q.contains("size") || q.contains("dimension") || q.contains("fl") || q.contains("hc") || q.contains("ac") || q.contains("imt")
    }

    private static func asksAboutSegmentationOrSize(_ q: String) -> Bool {
        q.contains("mask") || q.contains("segmentation") || q.contains("lesion") || q.contains("area") || q.contains("volume") || q.contains("bbox") || q.contains("bounding")
    }

    private static func asksAboutQualityOrConfidence(_ q: String) -> Bool {
        q.contains("quality") || q.contains("confidence") || q.contains("certain")
    }

    private static func asksAboutNormality(_ q: String) -> Bool {
        q.contains("normal") || q.contains("abnormal") || q.hasPrefix("is this ") || q.hasPrefix("does this ")
    }

    private static func asksAboutPresenceAbsence(_ q: String) -> Bool {
        q.contains("is there")
            || q.contains("are there")
            || q.contains("do you see")
            || q.contains("does it show")
            || q.contains("present")
            || q.contains("absent")
            || q.contains("contains")
            || q.contains("have ")
            || q.contains("has ")
    }

    private static func asksForImpression(_ q: String) -> Bool {
        q.contains("impression") || q.contains("summary") || q.contains("report")
    }

    private static func asksWhyOrReasoning(_ q: String) -> Bool {
        q.contains("why")
            || q.contains("reason")
            || q.contains("reasoning")
            || q.contains("how do you know")
            || q.contains("what supports")
            || q.contains("what evidence")
            || q.contains("why selected")
            || q.contains("why was")
    }

    private static func asksForClassSelection(_ q: String) -> Bool {
        q.contains("selected")
            || q.contains("classified")
            || q.contains("class")
            || q.contains("predicted")
            || q.contains("label")
    }

    private static func asksForPureVisualDescription(_ q: String) -> Bool {
        q.contains("describe the image")
            || q.contains("describe this image")
            || q.contains("what is visible")
            || q.contains("visual description")
            || q.contains("echotexture")
            || q.contains("borders")
            || q.contains("symmetry")
            || q.contains("appearance")
    }

    private static func asksForImageOnlyDescription(_ q: String) -> Bool {
        (q.contains("from the image alone") || q.contains("image only"))
            && (q.contains("describe") || q.contains("what do you see") || q.contains("what is visible"))
    }

    private static func presenceAbsenceAnswer(for q: String, report: PULSEAnalysisReport) -> String? {
        if q.contains("lesion") || q.contains("mass") || q.contains("nodule") || q.contains("cyst") {
            if let segmentation = report.specialistFindings.first(where: { $0.category == .segmentation }) {
                let details = segmentation.displayDetails.prefix(3).joined(separator: "; ")
                if details.isEmpty == false {
                    return "Yes. A lesion-focused segmentation output is present on this image with these recorded extent details: \(details)."
                }
                return "Yes. A lesion-focused segmentation output is present on this image."
            }
        }

        if let pathology = primaryPathologyFinding(in: report), let label = pathology.label?.pulseDisplayText.lowercased() {
            let positive = normalizedLabelDoesSupportPositive(question: q, label: label)
            if let positive {
                var answer = directBinaryLead(question: q, positive: positive)
                answer += "The strongest structured assessment favors \(label)"
                if let confidence = pathology.confidencePercentText {
                    answer += " at \(confidence)"
                }
                answer += "."
                if let rationale = PULSEReasoningService.structuredSelectionRationale(for: report) {
                    answer += " \(rationale)"
                }
                return answer
            }
        }

        return nil
    }

    private static func primaryPathologyFinding(in report: PULSEAnalysisReport) -> PULSEFinding? {
        report.specialistFindings.first { finding in
            let task = finding.taskID.lowercased()
            return task.contains("classification")
                && !task.contains("domain")
                && !task.contains("view")
                && !task.contains("subview")
                && !task.contains("organ")
                && !task.contains("plane")
        }
    }

    private static func directBinaryLead(question: String, positive: Bool?) -> String {
        guard asksBinaryQuestion(question) else { return "" }
        guard let positive else { return "The available structured outputs do not support a definitive yes-or-no answer. " }
        return positive ? "Yes. " : "No. "
    }

    private static func reasoningAnswer(question: String, report: PULSEAnalysisReport, imageSize: CGSize) -> String {
        var parts: [String] = []

        if let rationale = PULSEReasoningService.structuredSelectionRationale(for: report) {
            parts.append(rationale)
        }

        if let measurement = PULSEReasoningService.structuredMeasurementSummary(for: report) {
            parts.append(measurement)
        }

        if let segmentation = PULSEReasoningService.structuredSegmentationSummary(for: report) {
            parts.append(segmentation)
        }

        if parts.isEmpty {
            parts.append(PULSEReasoningService.structuredFindingsSummary(for: report, imageSize: imageSize))
        }

        if asksAboutNormality(question), let pathology = primaryPathologyFinding(in: report), let label = pathology.label?.pulseDisplayText.lowercased() {
            var prefix = "The strongest structured assessment favors \(label)"
            if let confidence = pathology.confidencePercentText {
                prefix += " at \(confidence)"
            }
            prefix += "."
            parts.insert(prefix, at: 0)
        }

        return parts.joined(separator: " ")
    }

    private static func asksBinaryQuestion(_ q: String) -> Bool {
        let prefixes = ["is ", "are ", "does ", "do ", "did ", "can ", "could ", "would ", "will ", "was ", "were ", "has ", "have ", "had "]
        return prefixes.contains { q.hasPrefix($0) }
    }

    private static func normalizedLabelDoesSupportPositive(question: String, label: String) -> Bool? {
        if question.contains("normal") {
            if label.contains("normal") || label.contains("benign") {
                return true
            }
            if label.contains("abnormal") || label.contains("malignant") {
                return false
            }
        }

        if question.contains("abnormal") {
            if label.contains("abnormal") || label.contains("malignant") {
                return true
            }
            if label.contains("normal") || label.contains("benign") {
                return false
            }
        }

        if question.contains("benign") {
            if label.contains("benign") {
                return true
            }
            if label.contains("malignant") {
                return false
            }
        }

        if question.contains("malignant") || question.contains("cancer") {
            if label.contains("malignant") {
                return true
            }
            if label.contains("benign") {
                return false
            }
        }

        if question.contains("lesion") || question.contains("mass") || question.contains("nodule") || question.contains("cyst") {
            if label.contains("abnormal") || label.contains("malignant") || label.contains("benign") || label.contains("lesion") || label.contains("mass") || label.contains("nodule") || label.contains("cyst") {
                return true
            }
        }

        return nil
    }
}
