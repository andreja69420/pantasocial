import type { NextApiRequest, NextApiResponse } from 'next'
import nodemailer from 'nodemailer'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { name, business, email, phone, service, message } = req.body

  if (!name || !business || !email || !message) {
    return res.status(400).json({ error: 'Missing required fields' })
  }

  const transporter = nodemailer.createTransport({
    host:   process.env.SMTP_HOST   || 'smtp.gmail.com',
    port:   Number(process.env.SMTP_PORT) || 587,
    secure: false,
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  })

  const html = `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0d0d1a; color: #fff; border-radius: 12px; overflow: hidden;">
      <div style="background: linear-gradient(135deg, #6366f1, #a855f7); padding: 24px 32px;">
        <h1 style="margin: 0; font-size: 22px; font-weight: 700;">New Contact Form Submission</h1>
        <p style="margin: 4px 0 0; opacity: 0.8; font-size: 14px;">PantaSocial LLC Website</p>
      </div>
      <div style="padding: 32px;">
        <table style="width: 100%; border-collapse: collapse;">
          <tr><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; width: 130px;">Name</td><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); font-weight: 600;">${name}</td></tr>
          <tr><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Business</td><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">${business}</td></tr>
          <tr><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Email</td><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08);"><a href="mailto:${email}" style="color: #a5bafc;">${email}</a></td></tr>
          ${phone ? `<tr><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Phone</td><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">${phone}</td></tr>` : ''}
          ${service ? `<tr><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Service</td><td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">${service}</td></tr>` : ''}
          <tr><td style="padding: 10px 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; vertical-align: top;">Message</td><td style="padding: 10px 0; line-height: 1.6;">${message.replace(/\n/g, '<br>')}</td></tr>
        </table>
      </div>
      <div style="padding: 16px 32px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; color: rgba(255,255,255,0.2);">
        Sent from pantasocial.com contact form · ${new Date().toLocaleString()}
      </div>
    </div>
  `

  try {
    await transporter.sendMail({
      from:    `"PantaSocial Website" <${process.env.SMTP_USER}>`,
      to:      'andreja@pantasocial.com',
      replyTo: email,
      subject: `New enquiry from ${name} — ${business}`,
      html,
    })
    return res.status(200).json({ success: true })
  } catch (err) {
    console.error('Mail send error:', err)
    return res.status(500).json({ error: 'Failed to send message. Please email us directly.' })
  }
}
