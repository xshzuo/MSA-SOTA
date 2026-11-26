#!/usr/bin/env swift
// Render the pages of a PDF to PNG for visual inspection.
// usage: swift pdfpages.swift <pdf> <outdir> [firstPage] [lastPage]
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: pdfpages.swift <pdf> <outdir> [first] [last]\n".data(using: .utf8)!)
    exit(2)
}
let pdfURL = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

guard let doc = CGPDFDocument(pdfURL as CFURL) else {
    FileHandle.standardError.write("cannot open the pdf\n".data(using: .utf8)!)
    exit(1)
}
let first = args.count > 3 ? (Int(args[3]) ?? 1) : 1
let last = args.count > 4 ? (Int(args[4]) ?? doc.numberOfPages) : doc.numberOfPages
print("pages: \(doc.numberOfPages)")
for i in first...min(last, doc.numberOfPages) {
    guard let page = doc.page(at: i) else { continue }
    let rect = page.getBoxRect(.mediaBox)
    let scale: CGFloat = 2.0
    let w = Int(rect.width * scale), h = Int(rect.height * scale)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.scaleBy(x: scale, y: scale)
    ctx.translateBy(x: -rect.origin.x, y: -rect.origin.y)
    ctx.drawPDFPage(page)
    guard let img = ctx.makeImage() else { continue }
    let out = outDir.appendingPathComponent(String(format: "page-%02d.png", i))
    guard let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil) else { continue }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
    print(out.path)
}
