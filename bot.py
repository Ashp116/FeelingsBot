import os
import discord
from discord import app_commands
from discord.ext import commands
from feelings import calculate
from dotenv import load_dotenv


load_dotenv()

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print("Bot is up and ready!")

    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.tree.command(name="analyze", description="Analyze your sentences and tells the emotion your sentence invokes.")
async def analyze(interaction: discord.Interaction, sentence: str):
    await interaction.response.send_message(f"The message expresses {calculate(sentence).lower()}!")

bot.run(os.environ['TOKEN'])