import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
    LinkPreviewOptions,
    BotCommand,
)
from aiogram.fsm.storage.memory import MemoryStorage


from ruamel.yaml import YAML
from pathlib import Path

yamlp = YAML(typ="safe")
conf = yamlp.load(
    Path(__file__).parent.joinpath("config.yaml").open("r", encoding="utf-8")
)

from pysondb import db

database = db.getDb(str(Path(__file__).parent.joinpath("database.json")))

dp = Dispatcher(storage=MemoryStorage())


class State(StatesGroup):
    selecting_admin_id_add = State()
    selecting_admin_id_rm = State()

    selecting_start_msg = State()
    selecting_autoreply_msg = State()
    selecting_new_vancancy_msg = State()

    add_new_channel = State()
    delete_channel = State()

    channel_selection = State()
    channel_selected = State()

    mailing_prompt = State()

    selecting_decline_reason = State()

    channel_selection_reply = State()
    channel_selected_reply = State()
    vacancy_accepted_reply = State()

    sent_to_moderation_reply = State()


@dp.message(Command("cancel"))
async def reset_state(message: Message, state: FSMContext) -> None:
    if not await state.get_state():
        return
    await state.clear()
    await message.answer(
        conf["messages"]["canceled"], reply_markup=ReplyKeyboardRemove()
    )


def hr_registred(rhr):
    hrs = database.getByQuery({"entity": "hr"})
    for hr in hrs:
        if str(hr["payload"]["id"]) == str(rhr):
            return True
    return False


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    if message.chat.id not in conf["superadmins"]:
        if not database.getByQuery(
            {"entity": "admin_receiver", "payload": str(message.chat.id)}
        ):
            if not hr_registred(message.chat.id):
                database.add({"entity": "hr", "payload": {"id": message.chat.id}})

    await message.answer(message.bot.greeting, parse_mode=ParseMode.MARKDOWN_V2)


def make_row_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True)


def make_col_keyboard(
    items: list[str], max_col: int = 3, first: bool = False
) -> ReplyKeyboardMarkup:
    keyboard = []

    if first:
        first = items.pop(0)
        keyboard.append([KeyboardButton(text=first)])

    last = items.pop()
    for i in range(0, len(items) - 1, max_col):
        keyboard.append(
            [
                KeyboardButton(text=items[i + k]) if len(items) > i + k else None
                for k in range(max_col)
            ]
        )
    keyboard = [[btn for btn in row if btn] for row in keyboard]
    keyboard.append([KeyboardButton(text=last)])

    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


@dp.message(Command("new_admin"))
async def new_admin_receiver(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.selecting_admin_id_add)
    await message.answer(
        "Введите ID юзера, который будет получать вакансии",
        reply_markup=make_row_keyboard(["/cancel"]),
    )


@dp.message(State.selecting_admin_id_add)
async def new_admin_receiver_id(message: Message, state: FSMContext) -> None:
    if row := database.getByQuery(
        {"entity": "admin_receiver", "payload": str(message.text)}
    ):
        await message.answer("Этот айди уже в списке, введите другой")
        return

    try:
        msg = await message.bot.send_message(
            message.text,
            "Вас назначили на роль админа-получателя вакансий",
            parse_mode=ParseMode.HTML,
        )
    except:
        await message.answer("Invalid ID, try again")
        return

    database.add({"entity": "admin_receiver", "payload": message.text})
    hr_record = database.getByQuery({"entity": "hr", "payload": {"id": msg.chat.id}})[0]
    database.deleteById(hr_record["id"])

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("delete_admin"))
async def rm_admin_receiver(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.selecting_admin_id_rm)
    await message.answer(
        "Введите ID юзера, которого хотите удалить из получателей",
        reply_markup=make_row_keyboard(["/cancel"]),
    )


@dp.message(State.selecting_admin_id_rm)
async def rm_admin_receiver_id(message: Message, state: FSMContext) -> None:
    try:
        await message.bot.get_chat(message.text)
    except:
        await message.answer("Invalid ID, try again")
        return

    if not (
        row := database.getByQuery(
            {"entity": "admin_receiver", "payload": str(message.text)}
        )
    ):
        await message.answer("This user is not one of receiving administrators")
        return

    database.deleteById(row[0]["id"])

    await message.answer(conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML)
    await state.clear()


@dp.message(Command("start_template"))
async def new_start_message(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.selecting_start_msg)
    await message.answer(
        "Пришлите новое сообщение для команды /start",
        reply_markup=make_row_keyboard(["/cancel"]),
    )


@dp.message(State.selecting_start_msg)
async def new_start_message_ack(message: Message, state: FSMContext) -> None:
    new_greeting = message.md_text

    if database.getByQuery({"entity": "greeting"}):
        database.updateByQuery({"entity": "greeting"}, {"payload": new_greeting})
    else:
        database.add({"entity": "greeting", "payload": new_greeting})

    message.bot.greeting = new_greeting

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("channel_selection_reply"))
async def channel_selection_reply(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.channel_selection_reply)
    await message.answer(
        "Пришлите новое сообщение для отправки после выбора канала в команде /new_vacancy\n"
        "Вы можете использовать заполнитель {channel_name} для модификации текста\n"
        "Пример: 'Вы выбрали {channel_name}, подтвердите выбор или измените его кнопкой ниже'",
        reply_markup=make_row_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.channel_selection_reply)
async def channel_selection_reply_ack(message: Message, state: FSMContext) -> None:
    new_channel_selection_reply = message.html_text

    if database.getByQuery({"entity": "channel_selection_reply"}):
        database.updateByQuery(
            {"entity": "channel_selection_reply"},
            {"payload": new_channel_selection_reply},
        )
    else:
        database.add(
            {
                "entity": "channel_selection_reply",
                "payload": new_channel_selection_reply,
            }
        )

    message.bot.channel_selection_reply = new_channel_selection_reply

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("channel_selected_reply"))
async def channel_selected_reply(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.channel_selected_reply)
    await message.answer(
        "Пришлите новое сообщение для отправки после выбора <b>и подтверждения</b> канала в команде /new_vacancy\n"
        "Пример: 'Отправьте текст вакансии'",
        reply_markup=make_row_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.channel_selected_reply)
async def channel_selected_reply_ack(message: Message, state: FSMContext) -> None:
    new_channel_selected_reply = message.html_text

    if database.getByQuery({"entity": "channel_selected_reply"}):
        database.updateByQuery(
            {"entity": "channel_selected_reply"},
            {"payload": new_channel_selected_reply},
        )
    else:
        database.add(
            {"entity": "channel_selected_reply", "payload": new_channel_selected_reply}
        )

    message.bot.channel_selected_reply = new_channel_selected_reply

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("vacancy_accepted_reply"))
async def vacancy_accepted_reply(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.vacancy_accepted_reply)
    await message.answer(
        "Пришлите новое сообщение для отправки после принятия вакансии в работу\n"
        "Вы можете использовать заполнитель {vacancy_description} для модификации текста\n"
        "Пример: 'Ваша вакансия {vacancy_description} принята модератором и скоро будет опубликована.'",
        reply_markup=make_row_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.vacancy_accepted_reply)
async def vacancy_accepted_reply_ack(message: Message, state: FSMContext) -> None:
    new_vacancy_accepted_reply = message.html_text

    if database.getByQuery({"entity": "vacancy_accepted_reply"}):
        database.updateByQuery(
            {"entity": "vacancy_accepted_reply"},
            {"payload": new_vacancy_accepted_reply},
        )
    else:
        database.add(
            {"entity": "vacancy_accepted_reply", "payload": new_vacancy_accepted_reply}
        )

    message.bot.vacancy_accepted_reply = new_vacancy_accepted_reply

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("sent_to_moderation_reply"))
async def vacancy_accepted_reply(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.sent_to_moderation_reply)
    await message.answer(
        "Пришлите новое сообщение для отправки после принятия на модерацию\n"
        "Пример: 'Спасибо, после модерации вакансия будет опубликована, чтобы предложить еще вакансию, напишите /new_vacancy'",
        reply_markup=make_row_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.sent_to_moderation_reply)
async def vacancy_accepted_reply_ack(message: Message, state: FSMContext) -> None:
    new_sent_to_moderation_reply = message.html_text

    if database.getByQuery({"entity": "sent_to_moderation_reply"}):
        database.updateByQuery(
            {"entity": "sent_to_moderation_reply"},
            {"payload": new_sent_to_moderation_reply},
        )
    else:
        database.add(
            {
                "entity": "sent_to_moderation_reply",
                "payload": new_sent_to_moderation_reply,
            }
        )

    message.bot.sent_to_moderation_reply = new_sent_to_moderation_reply

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("auto_reply"))
async def new_autoreply_message(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.selecting_autoreply_msg)
    await message.answer(
        "Пришлите новое сообщение для автоответа",
        reply_markup=make_row_keyboard(["/cancel"]),
    )


@dp.message(State.selecting_autoreply_msg)
async def new_autoreply_message_ack(message: Message, state: FSMContext) -> None:
    new_autoreply = message.md_text

    if database.getByQuery({"entity": "autoreply"}):
        database.updateByQuery({"entity": "autoreply"}, {"payload": new_autoreply})
    else:
        database.add({"entity": "autoreply", "payload": new_autoreply})

    message.bot.autoreply_msg = new_autoreply

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("new_vacancy_template"))
async def new_vacancy_message(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.selecting_new_vancancy_msg)
    await message.answer(
        "Отправьте новое сообщение для команды /new_vancancy",
        reply_markup=make_row_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.selecting_new_vancancy_msg)
async def new_vacancy_message_ack(message: Message, state: FSMContext) -> None:
    new_new_vancancy_msg = message.text

    if database.getByQuery({"entity": "new_vacancy_msg"}):
        database.updateByQuery(
            {"entity": "new_vacancy_msg"}, {"payload": new_new_vancancy_msg}
        )
    else:
        database.add({"entity": "new_vacancy_msg", "payload": new_new_vancancy_msg})

    message.bot.new_vacancy_msg = new_new_vancancy_msg

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("new_vacancy"))
async def new_vacancy(message: Message, state: FSMContext) -> None:
    channels = [
        ch["payload"]["sname"] for ch in database.getByQuery({"entity": "channel"})
    ]

    await message.answer(
        message.bot.new_vacancy_msg,
        reply_markup=make_col_keyboard([*channels, "/cancel"]),
    )
    await state.set_state(State.channel_selection)


@dp.message(Command("new_mailing"))
async def new_mailing(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    hrs = database.getByQuery({"entity": "hr"})

    await message.answer(
        f"Введите текст сообщения. Количество человек которым оно поступит: {len(hrs)}",
        reply_markup=make_col_keyboard(["/cancel"]),
        parse_mode=ParseMode.HTML,
    )
    await state.set_state(State.mailing_prompt)


@dp.message(State.mailing_prompt)
async def new_mailing_ack(message: Message, state: FSMContext) -> None:
    hrs = database.getByQuery({"entity": "hr"})
    await message.answer(
        f"Начинаю рассылку на {len(hrs)} чел.",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )

    for hr in hrs:
        try:
            await message.bot.send_message(
                hr["payload"]["id"], message.html_text, parse_mode=ParseMode.HTML
            )
        except:
            pass

        await asyncio.sleep(1)


@dp.message(State.channel_selection)
async def new_vacancy_ack0(message: Message, state: FSMContext) -> None:
    if "Оставить" in message.text:
        await state.set_state(State.channel_selected)
        await message.answer(
            message.bot.channel_selected_reply,
            reply_markup=make_col_keyboard(["/cancel"]),
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        channel_name = message.text
    except:
        await message.answer("Ошибка в формате сообщения, попробуйте выбрать кнопкой")
        return

    for channel in database.getByQuery({"entity": "channel"}):
        if channel_name in channel["payload"].values():
            await state.set_data({"channel_name": channel_name})
            channels = [
                ch["payload"]["sname"]
                for ch in database.getByQuery({"entity": "channel"})
            ]
            await message.answer(
                message.bot.channel_selection_reply.format(channel_name=channel_name),
                reply_markup=make_col_keyboard(
                    [f"Оставить {channel_name}", *channels], first=True
                ),
                parse_mode=ParseMode.HTML,
            )
            return

    await message.answer("Неизвестный канал, попробуйте выбрать кнопкой")


@dp.message(State.channel_selected)
async def new_vacancy_ack1(message: Message, state: FSMContext) -> None:
    admin_list = database.getByQuery({"entity": "admin_receiver"})
    fr_msgs = []
    channel_name = (await state.get_data())["channel_name"]

    for admin in admin_list:
        try:
            admin_chat = await message.bot.get_chat(admin["payload"])
        except Exception as e:
            print("Не удалось связатся с админом id=", admin["payload"])

        btn_ok = InlineKeyboardButton(
            text="Принять в работу", callback_data=f"respond:{message.message_id}:ok"
        )
        btn_decline = InlineKeyboardButton(
            text="Отклонить", callback_data=f"respond:{message.message_id}:decline"
        )

        kb = InlineKeyboardMarkup(inline_keyboard=[[btn_ok, btn_decline]])
        fr_msg = await message.bot.send_message(
            admin_chat.id,
            f"Новая вакансия для канала {channel_name!r}\n\n" + message.md_text,
            reply_markup=kb,
        )
        fr_msgs.append((fr_msg.chat.id, fr_msg.message_id))

    database.add(
        {
            "entity": "vacancy",
            "payload": {
                "msg_id": message.message_id,
                "msg_src": message.from_user.id,
                "frs": fr_msgs,
                "desc": message.html_text[:100].replace("\n", " "),
                "channel": channel_name,
            },
        }
    )

    await message.answer(
        message.bot.sent_to_moderation_reply,
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )
    await state.clear()


@dp.edited_message()
async def msg_edit(message: Message, state: FSMContext) -> None:
    for vc in database.getByQuery({"entity": "vacancy"}):
        if vc["payload"]["msg_id"] == message.message_id:
            for fchat, fmsg in vc["payload"]["frs"]:
                channel_name = vc["payload"]["channel"]
                btn_ok = InlineKeyboardButton(
                    text="Принять в работу",
                    callback_data=f"respond:{message.message_id}:ok",
                )
                btn_decline = InlineKeyboardButton(
                    text="Отклонить",
                    callback_data=f"respond:{message.message_id}:decline",
                )

                kb = InlineKeyboardMarkup(inline_keyboard=[[btn_ok, btn_decline]])
                await message.bot.edit_message_text(
                    f"Новая вакансия для канала {channel_name!r}\n\n" + message.md_text,
                    fchat,
                    fmsg,
                    reply_markup=kb,
                )


@dp.message(Command("add_new_channel"))
async def add_new_channel(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.add_new_channel)
    await message.answer(
        "Пример:\n" "'Питон Новый Канал - питон - t.me/durov'",
        reply_markup=make_row_keyboard(["/cancel"]),
        link_preview_options=LinkPreviewOptions(is_disabled=True),
        parse_mode=ParseMode.HTML,
    )


@dp.message(State.add_new_channel)
async def add_new_channel_ack(message: Message, state: FSMContext) -> None:
    try:
        name, sname, link = message.text.split(" - ", 3)
    except:
        await message.answer("Ошибка в формате сообщения, попробуйте еще раз")
        return

    channel = {"name": name, "sname": sname, "link": link}
    database.add({"entity": "channel", "payload": channel})

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message(Command("delete_channel"))
async def delete_channel(message: Message, state: FSMContext) -> None:
    if message.chat.id not in conf["superadmins"]:
        await message.answer("unauthorized")
        return

    await state.set_state(State.delete_channel)
    await message.answer(
        "Отправьте ссылку на канал или назвнание канала который нужно убрать",
        reply_markup=make_row_keyboard(["/cancel"]),
    )


@dp.message(State.delete_channel)
async def delete_channel_ack(message: Message, state: FSMContext) -> None:

    for channel in database.getByQuery({"entity": "channel"}):
        if message.text in channel["payload"].values():
            database.deleteById(channel["id"])
            await message.answer(
                conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
            )
            await state.clear()
            return

    await message.answer("Не удалось найти", reply_markup=ReplyKeyboardRemove())
    await state.clear()


@dp.callback_query()
async def process_callback_button1(callback_query: CallbackQuery, state: State):
    msg_id, status = callback_query.data.split(":")[1:]
    await callback_query.answer("Successful")

    for vacancy in database.getByQuery({"entity": "vacancy"}):
        if str(vacancy["payload"]["msg_id"]) == str(msg_id):
            if status == "ok":
                await callback_query.bot.send_message(
                    vacancy["payload"]["msg_src"],
                    callback_query.bot.vacancy_accepted_reply.format(
                        vacancy_description=vacancy["payload"]["desc"]
                    ),
                    parse_mode=ParseMode.HTML,
                )
                await state.clear()
                for fchat, fmsg in vacancy["payload"]["frs"]:
                    await callback_query.bot.edit_message_reply_markup(
                        fchat, fmsg, reply_markup=None
                    )
            else:
                await callback_query.message.answer(
                    "Введите причину отказа",
                    reply_markup=make_col_keyboard(["Отказать без причины"]),
                )
                await state.set_state(State.selecting_decline_reason)
                await state.set_data({"vacancy": vacancy})


@dp.message(State.selecting_decline_reason)
async def callback_decline_ack(message: Message, state: FSMContext) -> None:
    vacancy = (await state.get_data())["vacancy"]

    if message.text == "Отказать без причины":
        await message.bot.send_message(
            vacancy["payload"]["msg_src"],
            f"Ваша вакансия {vacancy['payload']['desc']!r} не принята модератором.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.bot.send_message(
            vacancy["payload"]["msg_src"],
            f"Ваша вакансия {vacancy['payload']['desc']!r} не принята модератором по причине: {message.text}.",
            parse_mode=ParseMode.HTML,
        )

    for fchat, fmsg in vacancy["payload"]["frs"]:
        await message.bot.edit_message_reply_markup(fchat, fmsg, reply_markup=None)

    await message.answer(
        conf["messages"]["successful"], reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML
    )
    await state.clear()


@dp.message()
async def autoreply(message: Message) -> None:
    await message.answer(message.bot.autoreply_msg, parse_mode=ParseMode.MARKDOWN_V2)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(
        token=conf["bot_token"],
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2),
    )

    await bot.set_my_commands(
        [
            BotCommand(command="/start", description="start"),
            # BotCommand(command="/start_template", description="set new greeting"),
            BotCommand(command="/new_vacancy", description="new vacancy"),
            # BotCommand(command="/add_new_channel", description="add new channel"),
            # BotCommand(command="/delete_channel", description="delete channel"),
            # BotCommand(command="/new_mailing", description="new mailing"),
            # BotCommand(command="/delete_admin", description="delete admin"),
            # BotCommand(command="/new_admin", description="new admin"),
            # BotCommand(command="/auto_reply", description="auto reply"),
            # BotCommand(
            #     command="/new_vacancy_template", description="new vacancy template"
            # ),
        ]
    )

    greeting_record = database.getByQuery({"entity": "greeting"})
    greeting = greeting_record[0].get("payload", None) if greeting_record else None
    bot.greeting = greeting or conf["greeting"]

    new_vacancy_record = database.getByQuery({"entity": "new_vacancy_msg"})
    new_vacancy_msg = (
        new_vacancy_record[0].get("payload", None) if new_vacancy_record else None
    )

    bot.new_vacancy_msg = new_vacancy_msg or "выберите канал"

    autoreply_record = database.getByQuery({"entity": "autoreply"})
    autoreply_msg = (
        autoreply_record[0].get("payload", None) if autoreply_record else None
    )

    bot.autoreply_msg = autoreply_msg or "не задано"

    channel_selection_reply_record = database.getByQuery(
        {"entity": "channel_selection_reply"}
    )
    channel_selection_reply = (
        channel_selection_reply_record[0].get("payload", None)
        if channel_selection_reply_record
        else None
    )

    bot.channel_selection_reply = (
        channel_selection_reply
        or "Вы выбрали {channel_name!r}, оставьте его и пришлите текст вакансии, после модерации мы ее опубликуем, либо выберите другой канал"
    )

    channel_selected_reply_record = database.getByQuery(
        {"entity": "channel_selected_reply"}
    )
    channel_selected_reply = (
        channel_selected_reply_record[0].get("payload", None)
        if channel_selected_reply_record
        else None
    )

    bot.channel_selected_reply = (
        channel_selected_reply or "Отправьте текст вакансии для публикации в канале"
    )

    vacancy_accepted_reply_record = database.getByQuery(
        {"entity": "vacancy_accepted_reply"}
    )
    vacancy_accepted_reply = (
        vacancy_accepted_reply_record[0].get("payload", None)
        if vacancy_accepted_reply_record
        else None
    )

    bot.vacancy_accepted_reply = (
        vacancy_accepted_reply
        or "Ваша вакансия {vacancy_description} принята модератором и скоро будет опубликована."
    )

    sent_to_moderation_record = database.getByQuery(
        {"entity": "sent_to_moderation_reply"}
    )
    sent_to_moderation_reply = (
        sent_to_moderation_record[0].get("payload", None)
        if sent_to_moderation_record
        else None
    )

    bot.sent_to_moderation_reply = (
        sent_to_moderation_reply
        or "Спасибо, после модерации вакансия будет опубликована, чтобы предложить еще вакансию, напишите /new_vacancy"
    )

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
