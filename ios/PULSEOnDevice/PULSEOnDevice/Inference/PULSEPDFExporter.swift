import Foundation
import UIKit

enum PULSEPDFExporter {
    static func export(analysis: PULSESavedAnalysis, image: UIImage?) throws -> URL {
        let exportsDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("PULSEPDF", isDirectory: true)
        try FileManager.default.createDirectory(at: exportsDirectory, withIntermediateDirectories: true, attributes: nil)

        let filename = sanitizedFilename("\(analysis.displaySourceName)_\(analysis.createdAt.ISO8601Format())") + ".pdf"
        let outputURL = exportsDirectory.appendingPathComponent(filename)

        let bounds = CGRect(x: 0, y: 0, width: 612, height: 792)
        let renderer = UIGraphicsPDFRenderer(bounds: bounds)

        try renderer.writePDF(to: outputURL) { context in
            var cursorY: CGFloat = 0
            var pageNumber = 0

            func beginPage() {
                context.beginPage()
                pageNumber += 1
                pdfBackground.setFill()
                UIBezierPath(rect: context.pdfContextBounds).fill()
                drawHeader(in: context.pdfContextBounds, analysis: analysis)
                drawFooter(in: context.pdfContextBounds, pageNumber: pageNumber)
                cursorY = 156
            }

            func ensureSpace(_ height: CGFloat) {
                if cursorY + height > context.pdfContextBounds.height - 42 {
                    beginPage()
                }
            }

            func drawSectionHeading(
                eyebrow: String,
                title: String,
                subtitle: String,
                accent: UIColor = pdfAccent
            ) {
                let eyebrowHeight = measuredHeight(
                    eyebrow.uppercased(),
                    font: roundedFont(size: 10.5, weight: .bold),
                    width: 540,
                    lineSpacing: 0,
                    kern: 1.1
                )
                let titleHeight = measuredHeight(title, font: roundedFont(size: 20, weight: .semibold), width: 540)
                let subtitleHeight = measuredHeight(subtitle, font: systemFont(size: 12.5, weight: .regular), width: 540, lineSpacing: 2)
                let blockHeight = eyebrowHeight + titleHeight + subtitleHeight + 12
                ensureSpace(blockHeight)

                drawText(
                    eyebrow.uppercased(),
                    in: CGRect(x: 36, y: cursorY, width: 540, height: eyebrowHeight),
                    font: roundedFont(size: 10.5, weight: .bold),
                    color: accent,
                    lineSpacing: 0,
                    kern: 1.1
                )
                drawText(
                    title,
                    in: CGRect(x: 36, y: cursorY + eyebrowHeight + 2, width: 540, height: titleHeight),
                    font: roundedFont(size: 20, weight: .semibold),
                    color: pdfNavy
                )
                drawText(
                    subtitle,
                    in: CGRect(x: 36, y: cursorY + eyebrowHeight + titleHeight + 6, width: 540, height: subtitleHeight),
                    font: systemFont(size: 12.5, weight: .regular),
                    color: pdfSecondary,
                    lineSpacing: 2
                )
                cursorY += blockHeight + 10
            }

            func drawSectionCard(
                eyebrow: String? = nil,
                title: String,
                bodyLines: [String],
                fill: UIColor,
                accent: UIColor = pdfAccent,
                emphasis: String? = nil
            ) {
                let bodyText = bodyLines.joined(separator: "\n")
                let eyebrowHeight = eyebrow == nil
                    ? CGFloat.zero
                    : measuredHeight(
                        eyebrow!.uppercased(),
                        font: roundedFont(size: 10.5, weight: .bold),
                        width: 500,
                        lineSpacing: 0,
                        kern: 1.0
                    )
                let emphasisWidth = emphasis.map { max(64, measuredWidth($0, font: roundedFont(size: 10.5, weight: .bold)) + 22) } ?? 0
                let titleWidth = emphasis == nil ? CGFloat(500) : max(CGFloat(320), CGFloat(500) - emphasisWidth - 12)
                let titleHeight = measuredHeight(title, font: roundedFont(size: 18, weight: .semibold), width: titleWidth)
                let bodyHeight = measuredHeight(bodyText, font: systemFont(size: 12.75, weight: .regular), width: 500, lineSpacing: 3)
                let totalHeight = max(98, eyebrowHeight + titleHeight + bodyHeight + 52)
                ensureSpace(totalHeight)

                let rect = CGRect(x: 36, y: cursorY, width: 540, height: totalHeight)
                let card = UIBezierPath(roundedRect: rect, cornerRadius: 20)
                fill.setFill()
                card.fill()
                UIColor.white.withAlphaComponent(0.72).setStroke()
                card.lineWidth = 1
                card.stroke()

                let accentBar = CGRect(x: rect.minX, y: rect.minY, width: rect.width, height: 6)
                accent.setFill()
                UIBezierPath(roundedRect: accentBar, cornerRadius: 6).fill()

                let contentX = rect.minX + 22
                var contentY = rect.minY + 18

                if let eyebrow {
                    drawText(
                        eyebrow.uppercased(),
                        in: CGRect(x: contentX, y: contentY, width: 500, height: eyebrowHeight),
                        font: roundedFont(size: 10.5, weight: .bold),
                        color: accent,
                        lineSpacing: 0,
                        kern: 1.0
                    )
                    contentY += eyebrowHeight + 6
                }

                if let emphasis {
                    let pillRect = CGRect(x: rect.maxX - emphasisWidth - 20, y: rect.minY + 18, width: emphasisWidth, height: 24)
                    accent.withAlphaComponent(0.12).setFill()
                    UIBezierPath(roundedRect: pillRect, cornerRadius: 12).fill()
                    drawText(
                        emphasis,
                        in: pillRect.insetBy(dx: 10, dy: 5),
                        font: roundedFont(size: 10.5, weight: .bold),
                        color: accent,
                        lineSpacing: 0,
                        kern: 0.5
                    )
                }

                drawText(
                    title,
                    in: CGRect(x: contentX, y: contentY, width: titleWidth, height: titleHeight),
                    font: roundedFont(size: 18, weight: .semibold),
                    color: pdfNavy
                )
                drawText(
                    bodyText,
                    in: CGRect(x: contentX, y: contentY + titleHeight + 12, width: 500, height: rect.height - (contentY - rect.minY) - titleHeight - 24),
                    font: systemFont(size: 12.75, weight: .regular),
                    color: pdfBody,
                    lineSpacing: 3
                )
                cursorY = rect.maxY + 16
            }

            func drawImageBlock(title: String, image: UIImage, caption: String) {
                let drawSize = fittedSize(for: image.size, maxWidth: 476, maxHeight: 250)
                let blockHeight = drawSize.height + 96
                ensureSpace(blockHeight)

                let rect = CGRect(x: 36, y: cursorY, width: 540, height: blockHeight)
                let card = UIBezierPath(roundedRect: rect, cornerRadius: 20)
                UIColor.white.setFill()
                card.fill()
                UIColor.white.withAlphaComponent(0.72).setStroke()
                card.lineWidth = 1
                card.stroke()

                drawText(
                    title,
                    in: CGRect(x: rect.minX + 18, y: rect.minY + 14, width: 504, height: 22),
                    font: roundedFont(size: 16, weight: .semibold),
                    color: pdfNavy
                )

                let imagePanel = CGRect(x: rect.minX + 16, y: rect.minY + 42, width: 508, height: drawSize.height + 18)
                pdfImageWell.setFill()
                UIBezierPath(roundedRect: imagePanel, cornerRadius: 16).fill()

                let imageRect = CGRect(
                    x: imagePanel.midX - (drawSize.width / 2),
                    y: imagePanel.midY - (drawSize.height / 2),
                    width: drawSize.width,
                    height: drawSize.height
                )
                image.draw(in: imageRect)

                drawText(
                    caption,
                    in: CGRect(x: rect.minX + 18, y: imagePanel.maxY + 8, width: 504, height: 18),
                    font: systemFont(size: 11.5, weight: .medium),
                    color: pdfSecondary
                )
                cursorY = rect.maxY + 16
            }

            func drawCategoryHeader(_ category: PULSEFindingCategory) {
                let accent = accentColor(for: category)
                let height: CGFloat = 20
                ensureSpace(height + 8)
                drawText(
                    category.title.uppercased(),
                    in: CGRect(x: 36, y: cursorY, width: 540, height: height),
                    font: roundedFont(size: 11, weight: .bold),
                    color: accent,
                    lineSpacing: 0,
                    kern: 1.1
                )
                cursorY += height + 6
            }

            func drawFindingCard(_ finding: PULSEFinding) {
                let category = finding.category
                var lines = [finding.summary]
                if let label = finding.label {
                    lines.append("Result: \(label.pulseDisplayText)")
                }
                if !finding.displayDetails.isEmpty {
                    lines.append(contentsOf: finding.displayDetails)
                }
                drawSectionCard(
                    eyebrow: category.title,
                    title: finding.title,
                    bodyLines: lines,
                    fill: categoryFill(for: category),
                    accent: accentColor(for: category),
                    emphasis: finding.confidencePercentText
                )

                for artifact in finding.artifacts {
                    if let artifactImage = UIImage(data: artifact.pngData) {
                        drawImageBlock(title: artifact.caption, image: artifactImage, caption: finding.title)
                    }
                }
            }

            beginPage()

            drawSectionHeading(
                eyebrow: "Case Overview",
                title: "Primary Interpretation",
                subtitle: "A concise summary of the selected domain and the most relevant local interpretation."
            )
            drawSectionCard(
                eyebrow: "Summary",
                title: (analysis.report.subviewText ?? analysis.report.detectedLabel).pulseDisplayText,
                bodyLines: [
                    "Source: \(analysis.displaySourceName)",
                    "Detected domain: \(analysis.report.detectedDomain.pulseDisplayText)",
                    analysis.report.subviewText == nil
                        ? "Primary result: \(analysis.report.detectedLabel.pulseDisplayText)"
                        : "Subview: \(analysis.report.subviewText?.pulseDisplayText ?? analysis.report.detectedLabel.pulseDisplayText)",
                    "Quality: \(analysis.report.quality.qualityLabel.capitalized)",
                    "Created: \(analysis.createdAt.pulseTimestamp)"
                ],
                fill: UIColor(red: 0.95, green: 0.98, blue: 1.0, alpha: 1.0),
                emphasis: analysis.report.primaryConfidencePercentText
            )

            if analysis.prompt.isEmpty == false {
                drawSectionHeading(
                    eyebrow: "Clinical Context",
                    title: "Submitted Prompt",
                    subtitle: "Locally stored context provided alongside this study."
                )
                drawSectionCard(
                    eyebrow: "Prompt",
                    title: "Clinical Context",
                    bodyLines: [analysis.prompt],
                    fill: UIColor(red: 0.98, green: 0.99, blue: 1.0, alpha: 1.0)
                )
            }

            drawSectionHeading(
                eyebrow: "Interpretation",
                title: "Impression",
                subtitle: "A short synthesis generated from the available on-device findings."
            )
            drawSectionCard(
                eyebrow: "Impression",
                title: "Clinical Impression",
                bodyLines: [analysis.report.impressionText, analysis.report.note],
                fill: UIColor(red: 0.94, green: 0.98, blue: 0.97, alpha: 1.0),
                accent: UIColor(red: 0.0, green: 0.52, blue: 0.48, alpha: 1.0)
            )

            if let reasonedReport = analysis.report.reasonedReport {
                drawSectionHeading(
                    eyebrow: "On-Device Reasoning",
                    title: "Clinician-Facing Interpretation",
                    subtitle: "A local reasoning layer generated from the structured findings and supporting evidence on the device."
                )
                let paragraphs = reasonedReport.output
                    .components(separatedBy: "\n\n")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                drawSectionCard(
                    eyebrow: reasonedReport.model,
                    title: "Reasoned Report",
                    bodyLines: paragraphs.isEmpty ? [reasonedReport.output] : paragraphs,
                    fill: UIColor(red: 0.96, green: 0.985, blue: 1.0, alpha: 1.0),
                    accent: UIColor(red: 0.04, green: 0.44, blue: 0.62, alpha: 1.0)
                )
            }

            if let image {
                drawSectionHeading(
                    eyebrow: "Source Image",
                    title: "Input Ultrasound",
                    subtitle: "The original frame used for local analysis and visual evidence rendering."
                )
                drawImageBlock(
                    title: "Input Ultrasound",
                    image: image,
                    caption: "\(Int(image.size.width)) × \(Int(image.size.height)) pixels"
                )
            }

            drawSectionHeading(
                eyebrow: "Quality",
                title: "AI Triage",
                subtitle: "Input quality indicators derived directly from the acquired frame."
            )
            drawSectionCard(
                eyebrow: "Triage",
                title: "AI Triage",
                bodyLines: [
                    "Brightness: \(format(analysis.report.quality.brightness))",
                    "Contrast: \(format(analysis.report.quality.contrast))",
                    "Quality label: \(analysis.report.quality.qualityLabel.capitalized)"
                ],
                fill: UIColor(red: 0.99, green: 0.99, blue: 1.0, alpha: 1.0),
                accent: accentColor(forQualityLabel: analysis.report.quality.qualityLabel),
                emphasis: analysis.report.quality.qualityLabel.capitalized
            )

            if let routing = analysis.report.routingFinding {
                drawSectionHeading(
                    eyebrow: "Domain Selection",
                    title: "Selected Workflow",
                    subtitle: "The domain selection used to activate the relevant local interpretation path."
                )
                drawFindingCard(routing)
            }

            if !analysis.report.groupedFindings.isEmpty {
                drawSectionHeading(
                    eyebrow: "Specialist Findings",
                    title: "Structured Outputs",
                    subtitle: "Classification, measurement, and visual evidence grouped into a small set of review categories."
                )
                for (category, findings) in analysis.report.groupedFindings {
                    drawCategoryHeader(category)
                    for finding in findings {
                        drawFindingCard(finding)
                    }
                }
            }
        }

        return outputURL
    }

    private static func drawHeader(in bounds: CGRect, analysis: PULSESavedAnalysis) {
        let headerRect = CGRect(x: 24, y: 20, width: bounds.width - 48, height: 112)
        let gradient = CAGradientLayer()
        gradient.frame = headerRect
        gradient.colors = [
            UIColor(red: 0.03, green: 0.11, blue: 0.20, alpha: 1.0).cgColor,
            UIColor(red: 0.06, green: 0.42, blue: 0.58, alpha: 1.0).cgColor
        ]
        gradient.startPoint = CGPoint(x: 0, y: 0.5)
        gradient.endPoint = CGPoint(x: 1, y: 0.5)
        let image = UIGraphicsImageRenderer(size: headerRect.size).image { rendererContext in
            gradient.render(in: rendererContext.cgContext)
        }
        image.draw(in: headerRect)

        drawBrandSeal(in: CGRect(x: headerRect.maxX - 86, y: headerRect.minY + 18, width: 54, height: 54))

        let badgeRect = CGRect(x: headerRect.minX + 24, y: headerRect.minY + 20, width: 150, height: 24)
        UIColor.white.withAlphaComponent(0.16).setFill()
        UIBezierPath(roundedRect: badgeRect, cornerRadius: 12).fill()
        drawText(
            "PULSE ON-DEVICE",
            in: badgeRect.insetBy(dx: 10, dy: 4),
            font: roundedFont(size: 11, weight: .bold),
            color: .white,
            lineSpacing: 0,
            kern: 0.8
        )

        if let confidence = analysis.report.primaryConfidencePercentText {
            let confidenceWidth = max(64, measuredWidth(confidence, font: roundedFont(size: 10.5, weight: .bold)) + 22)
            let pillRect = CGRect(x: headerRect.maxX - confidenceWidth - 104, y: headerRect.minY + 20, width: confidenceWidth, height: 24)
            UIColor.white.withAlphaComponent(0.14).setFill()
            UIBezierPath(roundedRect: pillRect, cornerRadius: 12).fill()
            drawText(
                confidence,
                in: pillRect.insetBy(dx: 10, dy: 4),
                font: roundedFont(size: 10.5, weight: .bold),
                color: .white
            )
        }

        drawText(
            (analysis.report.subviewText ?? analysis.report.detectedLabel).pulseDisplayText,
            in: CGRect(x: headerRect.minX + 24, y: headerRect.minY + 48, width: 360, height: 32),
            font: roundedFont(size: 28, weight: .bold),
            color: .white
        )
        drawText(
            "\(analysis.report.detectedDomain.pulseDisplayText) domain • \(analysis.createdAt.pulseTimestamp)",
            in: CGRect(x: headerRect.minX + 24, y: headerRect.minY + 82, width: 360, height: 16),
            font: systemFont(size: 12.5, weight: .medium),
            color: UIColor.white.withAlphaComponent(0.86)
        )
        drawText(
            analysis.displaySourceName,
            in: CGRect(x: headerRect.minX + 24, y: headerRect.minY + 96, width: 360, height: 14),
            font: systemFont(size: 11, weight: .medium),
            color: UIColor.white.withAlphaComponent(0.72)
        )
    }

    private static func drawFooter(in bounds: CGRect, pageNumber: Int) {
        let footerY = bounds.height - 28
        let footerRect = CGRect(x: 36, y: footerY, width: bounds.width - 72, height: 16)
        drawText(
            "Generated on device by PULSE • Page \(pageNumber)",
            in: footerRect,
            font: systemFont(size: 10.5, weight: .medium),
            color: pdfSecondary
        )
    }

    private static func drawBrandSeal(in rect: CGRect) {
        let ctx = UIGraphicsGetCurrentContext()
        ctx?.saveGState()

        let outer = UIBezierPath(ovalIn: rect)
        UIColor(red: 0.05, green: 0.57, blue: 0.73, alpha: 0.24).setFill()
        outer.fill()

        let inner = rect.insetBy(dx: 6, dy: 6)
        UIBezierPath(ovalIn: inner).addClip()

        let gradient = CAGradientLayer()
        gradient.frame = inner
        gradient.colors = [
            UIColor(red: 0.04, green: 0.17, blue: 0.31, alpha: 1.0).cgColor,
            UIColor(red: 0.06, green: 0.50, blue: 0.72, alpha: 1.0).cgColor
        ]
        gradient.startPoint = CGPoint(x: 0, y: 0)
        gradient.endPoint = CGPoint(x: 1, y: 1)
        let rendered = UIGraphicsImageRenderer(size: inner.size).image { rendererContext in
            gradient.render(in: rendererContext.cgContext)
        }
        rendered.draw(in: inner)

        let cx = inner.midX
        let cy = inner.midY - 7
        UIColor(red: 0.48, green: 0.86, blue: 0.80, alpha: 1.0).setFill()
        UIBezierPath(roundedRect: CGRect(x: cx - 4, y: cy - 12, width: 8, height: 24), cornerRadius: 3).fill()
        UIBezierPath(roundedRect: CGRect(x: cx - 12, y: cy - 4, width: 24, height: 8), cornerRadius: 3).fill()

        let path = UIBezierPath()
        path.move(to: CGPoint(x: inner.minX + 10, y: inner.maxY - 17))
        path.addLine(to: CGPoint(x: inner.minX + 18, y: inner.maxY - 17))
        path.addLine(to: CGPoint(x: inner.midX - 3, y: inner.maxY - 28))
        path.addLine(to: CGPoint(x: inner.midX + 6, y: inner.maxY - 12))
        path.addLine(to: CGPoint(x: inner.maxX - 10, y: inner.maxY - 22))
        UIColor.white.setStroke()
        path.lineWidth = 3.4
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        path.stroke()

        ctx?.restoreGState()
    }

    private static func drawText(
        _ text: String,
        in rect: CGRect,
        font: UIFont,
        color: UIColor,
        lineSpacing: CGFloat = 0,
        kern: CGFloat = 0
    ) {
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineBreakMode = .byWordWrapping
        paragraph.lineSpacing = lineSpacing
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: color,
            .paragraphStyle: paragraph,
            .kern: kern
        ]
        NSString(string: text).draw(with: rect, options: [.usesLineFragmentOrigin, .usesFontLeading], attributes: attributes, context: nil)
    }

    private static func measuredHeight(
        _ text: String,
        font: UIFont,
        width: CGFloat,
        lineSpacing: CGFloat = 0,
        kern: CGFloat = 0
    ) -> CGFloat {
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineBreakMode = .byWordWrapping
        paragraph.lineSpacing = lineSpacing
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .paragraphStyle: paragraph,
            .kern: kern
        ]
        let rect = NSString(string: text).boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: attributes,
            context: nil
        )
        return ceil(rect.height)
    }

    private static func measuredWidth(_ text: String, font: UIFont) -> CGFloat {
        let rect = NSString(string: text).boundingRect(
            with: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font],
            context: nil
        )
        return ceil(rect.width)
    }

    private static func fittedSize(for size: CGSize, maxWidth: CGFloat, maxHeight: CGFloat) -> CGSize {
        guard size.width > 0, size.height > 0 else {
            return CGSize(width: maxWidth, height: maxHeight * 0.6)
        }
        let widthRatio = maxWidth / size.width
        let heightRatio = maxHeight / size.height
        let scale = min(widthRatio, heightRatio)
        return CGSize(width: size.width * scale, height: size.height * scale)
    }

    private static func sanitizedFilename(_ raw: String) -> String {
        let invalid = CharacterSet.alphanumerics.union(.init(charactersIn: "-_")).inverted
        return raw
            .components(separatedBy: invalid)
            .filter { !$0.isEmpty }
            .joined(separator: "_")
            .lowercased()
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private static func roundedFont(size: CGFloat, weight: UIFont.Weight) -> UIFont {
        let base = UIFont.systemFont(ofSize: size, weight: weight)
        guard let descriptor = base.fontDescriptor.withDesign(.rounded) else {
            return base
        }
        return UIFont(descriptor: descriptor, size: size)
    }

    private static func systemFont(size: CGFloat, weight: UIFont.Weight) -> UIFont {
        UIFont.systemFont(ofSize: size, weight: weight)
    }

    private static func accentColor(for category: PULSEFindingCategory) -> UIColor {
        switch category {
        case .routing, .segmentation, .multimodal:
            return pdfAccent
        case .classification:
            return pdfSuccess
        case .detection:
            return pdfWarning
        case .measurement:
            return pdfCritical
        case .other:
            return pdfSecondary
        }
    }

    private static func categoryFill(for category: PULSEFindingCategory) -> UIColor {
        switch category {
        case .routing, .segmentation, .multimodal:
            return UIColor(red: 0.97, green: 0.985, blue: 1.0, alpha: 1.0)
        case .classification:
            return UIColor(red: 0.97, green: 0.995, blue: 0.98, alpha: 1.0)
        case .detection:
            return UIColor(red: 1.0, green: 0.985, blue: 0.96, alpha: 1.0)
        case .measurement:
            return UIColor(red: 1.0, green: 0.975, blue: 0.975, alpha: 1.0)
        case .other:
            return UIColor(red: 0.99, green: 0.992, blue: 0.995, alpha: 1.0)
        }
    }

    private static func accentColor(forQualityLabel label: String) -> UIColor {
        switch label.lowercased() {
        case "good":
            return pdfSuccess
        case "medium":
            return pdfWarning
        case "low":
            return pdfCritical
        default:
            return pdfSecondary
        }
    }

    private static let pdfBackground = UIColor(red: 0.982, green: 0.988, blue: 0.995, alpha: 1.0)
    private static let pdfImageWell = UIColor(red: 0.964, green: 0.976, blue: 0.987, alpha: 1.0)
    private static let pdfNavy = UIColor(red: 0.05, green: 0.13, blue: 0.23, alpha: 1.0)
    private static let pdfAccent = UIColor(red: 0.04, green: 0.44, blue: 0.62, alpha: 1.0)
    private static let pdfSuccess = UIColor(red: 0.11, green: 0.55, blue: 0.39, alpha: 1.0)
    private static let pdfWarning = UIColor(red: 0.84, green: 0.57, blue: 0.18, alpha: 1.0)
    private static let pdfCritical = UIColor(red: 0.75, green: 0.29, blue: 0.26, alpha: 1.0)
    private static let pdfBody = UIColor(red: 0.22, green: 0.28, blue: 0.34, alpha: 1.0)
    private static let pdfSecondary = UIColor(red: 0.42, green: 0.48, blue: 0.54, alpha: 1.0)
}
